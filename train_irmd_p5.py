from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from irmd_qwen.data import Example, SequenceCorpus
from pretrain_p5_qwen import chat_prefix, evaluate, p5_prompt


TEMPLATES = ("2-1", "2-2", "2-3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IRMD with the exact P5-style Qwen item space as its recommendation head"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="existing autoregressive LoRA adapter to continue from")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--metadata")
    parser.add_argument("--corpus")
    parser.add_argument("--teacher-states")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-users", type=int, default=1000)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-train-users", type=int, default=0)
    parser.add_argument("--use-all-training-users", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--recommendation-weight", type=float, default=1.0)
    parser.add_argument("--reasoning-weight", type=float, default=1.0)
    parser.add_argument("--implicit-weight", type=float, default=0.7)
    parser.add_argument(
        "--reasoning-fraction",
        type=float,
        default=-1.0,
        help="fixed Full/Compressed/Implicit fraction in [0,1]; negative keeps cosine annealing",
    )
    parser.add_argument(
        "--implicit-distance", choices=["mse", "normalized_mse", "cosine"], default="mse"
    )
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-memory-fraction", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


class IRMDP5Dataset(Dataset):
    def __init__(
        self,
        corpus: SequenceCorpus,
        distillation_path: str | None,
        use_all_training_users: bool,
        max_train_users: int,
        seed: int,
    ) -> None:
        examples = list(corpus.training_examples(all_prefixes=False))
        example_by_id = {example.example_id: example for example in examples}
        raw_paths = []
        if distillation_path:
            raw_paths = [
                json.loads(line)
                for line in Path(distillation_path).read_text().splitlines()
                if line.strip()
            ]
        paths_by_id: dict[str, list[dict]] = {}
        for path in raw_paths:
            if path["example_id"] in example_by_id:
                paths_by_id.setdefault(path["example_id"], []).append(path)

        if max_train_users and len(examples) > max_train_users:
            distilled_ids = set(paths_by_id)
            distilled_examples = [example for example in examples if example.example_id in distilled_ids]
            other_examples = [example for example in examples if example.example_id not in distilled_ids]
            remaining = max(0, max_train_users - len(distilled_examples))
            examples = distilled_examples + random.Random(seed).sample(
                other_examples, min(remaining, len(other_examples))
            )
        selected_ids = {example.example_id for example in examples}
        rows: list[dict] = []
        if use_all_training_users:
            for example in examples:
                distilled = paths_by_id.get(example.example_id, [])
                if distilled:
                    for index, path in enumerate(distilled):
                        rows.append(
                            {
                                "example": example,
                                "template": TEMPLATES[index % len(TEMPLATES)],
                                "reasoning": path.get("reasoning", ""),
                                "state_index": path.get("state_index", -1),
                            }
                        )
                else:
                    for template in TEMPLATES:
                        rows.append(
                            {
                                "example": example,
                                "template": template,
                                "reasoning": "",
                                "state_index": -1,
                            }
                        )
        else:
            for path in raw_paths:
                if path["example_id"] not in selected_ids:
                    continue
                example = example_by_id[path["example_id"]]
                rows.append(
                    {
                        "example": example,
                        "template": TEMPLATES[len(rows) % len(TEMPLATES)],
                        "reasoning": path.get("reasoning", ""),
                        "state_index": path.get("state_index", -1),
                    }
                )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def pad_rows(rows: list[list[int]], pad: int, label_pad: int | None = None) -> torch.Tensor:
    length = max(map(len, rows))
    fill = pad if label_pad is None else label_pad
    return torch.tensor([row + [fill] * (length - len(row)) for row in rows], dtype=torch.long)


def collate_rows(tokenizer, rows: list[dict], anneal: float, max_length: int, device):
    prefixes, rec_ids, rec_labels, reason_ids, reason_labels, state_indices = [], [], [], [], [], []
    for row in rows:
        example: Example = row["example"]
        prefix = chat_prefix(tokenizer, p5_prompt(example, row["template"]))
        target = tokenizer(str(example.target), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        prefix = prefix[-max(1, max_length - len(target)) :]
        prefixes.append(prefix)
        rec_ids.append(prefix + target)
        rec_labels.append([-100] * len(prefix) + target)

        full_reasoning = tokenizer(row["reasoning"], add_special_tokens=False).input_ids
        keep = int(round(len(full_reasoning) * anneal))
        if anneal < 0.02:
            keep = 0
        if 0 < keep < len(full_reasoning):
            head = max(1, keep // 3)
            tail = keep - head
            reasoning = full_reasoning[:head] + (full_reasoning[-tail:] if tail else [])
        else:
            reasoning = full_reasoning[:keep]
        reasoning = reasoning[: max(0, max_length - len(prefix))]
        reason_ids.append(prefix + reasoning)
        reason_labels.append([-100] * len(prefix) + reasoning)
        state_indices.append(row["state_index"])

    prefix_ids = pad_rows(prefixes, tokenizer.pad_token_id).to(device)
    prefix_mask = prefix_ids.ne(tokenizer.pad_token_id).long()
    return {
        "prefix_ids": prefix_ids,
        "prefix_mask": prefix_mask,
        "rec_ids": pad_rows(rec_ids, tokenizer.pad_token_id).to(device),
        "rec_labels": pad_rows(rec_labels, tokenizer.pad_token_id, -100).to(device),
        "reason_ids": pad_rows(reason_ids, tokenizer.pad_token_id).to(device),
        "reason_labels": pad_rows(reason_labels, tokenizer.pad_token_id, -100).to(device),
        "state_indices": torch.tensor(state_indices, dtype=torch.long),
    }


def cosine_lambda(step: int, total_steps: int) -> float:
    progress = min(1.0, step / max(1, total_steps - 1))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def implicit_loss(student: torch.Tensor, teacher: torch.Tensor, mode: str) -> torch.Tensor:
    student, teacher = student.float(), teacher.float()
    if mode == "mse":
        return F.mse_loss(student, teacher)
    if mode == "normalized_mse":
        return F.mse_loss(F.normalize(student, dim=-1), F.normalize(teacher, dim=-1))
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def hidden_backbone(model):
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    return causal_lm.model


def main() -> None:
    args = parse_args()
    if not 0.0 < args.max_memory_fraction <= 1.0:
        raise ValueError("--max-memory-fraction must be in (0, 1]")
    if args.reasoning_fraction > 1.0:
        raise ValueError("--reasoning-fraction must not exceed 1")
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, 0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    corpus = SequenceCorpus.load(args.data_root, args.dataset, args.metadata)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = (
        PeftModel.from_pretrained(base_model, args.adapter, is_trainable=not args.evaluation_only)
        if args.adapter
        else base_model
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.cuda()
    device = model.device

    states = load_file(args.teacher_states)["teacher_states"] if args.teacher_states else None
    teacher_hidden = states.shape[1] if states is not None else model.config.hidden_size
    implicit_head = nn.Linear(teacher_hidden, model.config.hidden_size, bias=False).to(
        device=device, dtype=compute_dtype
    )

    if args.evaluation_only:
        test = evaluate(
            model,
            tokenizer,
            corpus,
            args.eval_batch_size,
            args.eval_max_users,
            args.num_beams,
            args.max_input_length,
            args.max_new_tokens,
        )
        result = {"mode": "evaluation_only", "test": test, "shared_item_space": "p5_numeric_trie"}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return

    dataset = IRMDP5Dataset(
        corpus,
        args.corpus,
        args.use_all_training_users,
        args.max_train_users,
        args.seed,
    )
    if not dataset:
        raise ValueError("no training rows; provide --corpus or --use-all-training-users")
    max_state_index = max(row["state_index"] for row in dataset.rows)
    if max_state_index >= 0 and states is None:
        raise ValueError("--teacher-states is required for distilled rows")
    if states is not None and max_state_index >= states.shape[0]:
        raise ValueError("teacher state indices do not match the distillation corpus")

    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=True, collate_fn=lambda x: x)
    # GradScaler cannot unscale gradients whose trainable parameters are themselves
    # FP16. Keep the small LoRA/implicit-head optimizer weights in FP32 on GPUs
    # without BF16 support; autocast still runs the forward pass in FP16.
    if compute_dtype == torch.float16:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        implicit_head.float()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(implicit_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=compute_dtype == torch.float16)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_steps = max(1, args.epochs * steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        implicit_head.train()
        for micro_step, rows in enumerate(loader):
            anneal = (
                args.reasoning_fraction
                if args.reasoning_fraction >= 0.0
                else cosine_lambda(global_step, total_steps)
            )
            batch = collate_rows(tokenizer, rows, anneal, args.max_input_length, device)
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                rec = model(
                    input_ids=batch["rec_ids"], labels=batch["rec_labels"], use_cache=False
                ).loss
                has_reasoning = batch["reason_labels"].ne(-100).any()
                if has_reasoning:
                    reason = model(
                        input_ids=batch["reason_ids"],
                        labels=batch["reason_labels"],
                        use_cache=False,
                    ).loss
                else:
                    reason = rec * 0.0
                distill_mask = batch["state_indices"].ge(0)
                if distill_mask.any():
                    hidden = hidden_backbone(model)(
                        input_ids=batch["prefix_ids"],
                        attention_mask=batch["prefix_mask"],
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
                    lengths = batch["prefix_mask"].sum(dim=1)
                    student_state = hidden[
                        torch.arange(hidden.shape[0], device=device), lengths - 1
                    ][distill_mask.to(device)]
                    teacher_state = states[batch["state_indices"][distill_mask]].to(
                        device=device, dtype=compute_dtype
                    )
                    implicit = implicit_loss(student_state, implicit_head(teacher_state), args.implicit_distance)
                else:
                    implicit = rec * 0.0
                loss = (
                    args.recommendation_weight * rec
                    + args.reasoning_weight * anneal * reason
                    + args.implicit_weight * (1.0 - anneal) * implicit
                )
            scaled_loss = loss / args.gradient_accumulation
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "total_steps": total_steps,
                            "lambda": anneal,
                            "loss": loss.item(),
                            "rec": rec.item(),
                            "reason": reason.item(),
                            "implicit": implicit.item(),
                        }
                    ),
                    flush=True,
                )

    if not args.no_save:
        model.save_pretrained(output / "model")
        tokenizer.save_pretrained(output / "model")
        torch.save(implicit_head.state_dict(), output / "implicit_head.pt")
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    test = evaluate(
        model,
        tokenizer,
        corpus,
        args.eval_batch_size,
        args.eval_max_users,
        args.num_beams,
        args.max_input_length,
        args.max_new_tokens,
    )
    result = {
        "student": args.model,
        "steps": global_step,
        "training_rows": len(dataset),
        "shared_item_space": "p5_numeric_trie",
        "test": test,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

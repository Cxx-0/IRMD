from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from irmd_qwen.data import Example, SequenceCorpus


SYSTEM_PROMPT = "You are an efficient sequential recommendation model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P5 sequential objectives adapted to a Qwen student")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-users", type=int, default=1000)
    parser.add_argument("--preflight-users", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-train-users", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=0, help="enable low-memory LoRA when greater than zero")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.0,
        help="cap this process's CUDA allocator to a fraction of total GPU memory; 0 disables the cap",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip model/tokenizer checkpoint saving (useful for short smoke tests).",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="load an existing model or PEFT adapter and skip optimizer/training",
    )
    return parser.parse_args()


def p5_prompt(example: Example, template: str) -> str:
    sequence = " -> ".join(str(item) for item in example.history)
    if template == "2-1":
        return (
            f"Given the following purchase history of user_{example.user_id} : \n {sequence} \n "
            "predict next possible item to be purchased by the user ?"
        )
    if template == "2-2":
        return (
            f"I find the purchase history list of user_{example.user_id} : \n {sequence} \n "
            "I wonder what is the next item to recommend to the user . Can you help me decide ?"
        )
    return (
        f"Here is the purchase history list of user_{example.user_id} : \n {sequence} \n "
        "try to recommend next item to the user"
    )


def chat_prefix(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(rendered, add_special_tokens=False).input_ids


class P5SequentialDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.rows = [
            (example, template)
            for example in examples
            for template in ("2-1", "2-2", "2-3")
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Example, str]:
        return self.rows[index]


@dataclass
class P5Collator:
    tokenizer: object
    max_input_length: int

    def __call__(self, rows: list[tuple[Example, str]]) -> dict[str, torch.Tensor]:
        sequences, labels = [], []
        for example, template in rows:
            prefix = chat_prefix(self.tokenizer, p5_prompt(example, template))
            answer = self.tokenizer(str(example.target), add_special_tokens=False).input_ids
            answer = answer + [self.tokenizer.eos_token_id]
            max_prefix = max(1, self.max_input_length - len(answer))
            ids = prefix[-max_prefix:] + answer
            sequences.append(ids)
            labels.append([-100] * (len(ids) - len(answer)) + answer)
        length = max(map(len, sequences))
        input_ids, attention, label_ids = [], [], []
        for ids, row_labels in zip(sequences, labels):
            padding = length - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * padding)
            attention.append([1] * len(ids) + [0] * padding)
            label_ids.append(row_labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def build_item_trie(tokenizer, num_items: int):
    trie: dict = {}
    lookup = {}
    for item_id in range(1, num_items):
        tokens = tuple(tokenizer(str(item_id), add_special_tokens=False).input_ids)
        lookup[tokens] = item_id
        node = trie
        for token in (*tokens, tokenizer.eos_token_id):
            node = node.setdefault(token, {})
    return trie, lookup


def update_metrics(totals: dict[str, float], ranking: list[int], target: int) -> None:
    for k in (10, 20):
        try:
            rank = ranking[:k].index(target) + 1
        except ValueError:
            continue
        totals[f"HR@{k}"] += 1.0
        totals[f"NDCG@{k}"] += 1.0 / math.log2(rank + 1)


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    corpus,
    batch_size,
    max_users,
    num_beams,
    max_input_length,
    max_new_tokens,
):
    model.eval()
    examples = list(corpus.evaluation_examples("test"))
    if max_users:
        indices = sorted(random.Random(43).sample(range(len(examples)), min(max_users, len(examples))))
        examples = [examples[index] for index in indices]
    trie, lookup = build_item_trie(tokenizer, corpus.num_items)
    totals = {"HR@10": 0.0, "NDCG@10": 0.0, "HR@20": 0.0, "NDCG@20": 0.0}
    valid = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p5_prompt(example, "2-3")},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in batch
        ]
        encoded = tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        input_length = encoded.input_ids.shape[1]

        def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            node = trie
            for token in input_ids[input_length:].tolist():
                if token not in node:
                    return [tokenizer.eos_token_id]
                node = node[token]
            return list(node) or [tokenizer.eos_token_id]

        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            early_stopping=True,
            prefix_allowed_tokens_fn=allowed_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for row, example in enumerate(batch):
            ranking = []
            sequences = outputs[row * num_beams : (row + 1) * num_beams, input_length:]
            for generated in sequences.tolist():
                if tokenizer.eos_token_id in generated:
                    generated = generated[: generated.index(tokenizer.eos_token_id)]
                item_id = lookup.get(tuple(generated))
                if item_id is not None and item_id not in ranking:
                    ranking.append(item_id)
                    valid += 1
            seen = set(example.history)
            ranking = [item_id for item_id in ranking if item_id not in seen]
            update_metrics(totals, ranking, example.target)
        print(json.dumps({"evaluated": min(start + len(batch), len(examples)), "total": len(examples)}), flush=True)
    tokenizer.padding_side = old_padding_side
    count = len(examples)
    result = {key: value / count for key, value in totals.items()}
    result["valid_unique_items_per_user"] = valid / count
    return result


def main() -> None:
    args = parse_args()
    if args.max_memory_fraction:
        if not 0.0 < args.max_memory_fraction <= 1.0:
            raise ValueError("--max-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, 0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    corpus = SequenceCorpus.load(args.data_root, args.dataset, args.metadata)
    examples = list(corpus.training_examples(all_prefixes=False))
    if args.max_train_users:
        examples = random.Random(args.seed).sample(examples, min(args.max_train_users, len(examples)))
    dataset = P5SequentialDataset(examples)
    print(json.dumps({"users": len(examples), "training_rows": len(dataset), "templates": ["2-1", "2-2", "2-3"]}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    # V100 full fine-tuning needs FP32 master parameters, but LoRA keeps the
    # frozen backbone in FP16 and therefore coexists with other GPU jobs.
    parameter_dtype = (
        compute_dtype
        if args.lora_r > 0 or args.eval_only
        else (torch.bfloat16 if bf16_supported else torch.float32)
    )
    print(
        json.dumps(
            {
                "cuda_compute_dtype": str(compute_dtype),
                "parameter_dtype": str(parameter_dtype),
            }
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=parameter_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if args.lora_r > 0:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.cuda()

    if args.preflight_users:
        preflight = evaluate(
            model,
            tokenizer,
            corpus,
            min(args.eval_batch_size, args.preflight_users),
            args.preflight_users,
            min(args.num_beams, 4),
            args.max_input_length,
            args.max_new_tokens,
        )
        print(json.dumps({"preflight": preflight}), flush=True)

    if args.eval_only:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
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
            "configuration": vars(args),
            "users": len(examples),
            "training_rows": len(dataset),
            "steps": 0,
            "test": test,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=P5Collator(tokenizer, args.max_input_length),
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not bf16_supported)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    global_step = 0
    running_loss = 0.0
    micro_count = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        for micro_step, batch in enumerate(loader):
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                loss = model(**batch, use_cache=False, return_dict=True).loss
            scaled_loss = loss / args.gradient_accumulation
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += loss.item()
            micro_count += 1
            should_step = (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "total_steps": total_steps,
                                "mean_loss": running_loss / micro_count,
                                "lr": scheduler.get_last_lr()[0],
                            }
                        ),
                        flush=True,
                    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_save:
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
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
        "configuration": vars(args),
        "users": len(examples),
        "training_rows": len(dataset),
        "steps": global_step,
        "test": test,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

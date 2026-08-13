from __future__ import annotations

import argparse
import json
import re
import zlib
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from irmd_qwen.data import SequenceCorpus
from irmd_qwen.prompts import SYSTEM_PROMPT, student_prompt, teacher_prompt


REASONING_RE = re.compile(
    r"\[REASONING\]\s*(.*?)(?=\s*\[(?:CONFIDENCE|PREDICTION)\]|\Z)", re.I | re.S
)
PREDICTION_RE = re.compile(r"\[PREDICTION\]\s*(?:\[?ITEM\s*)?(\d+)", re.I)
CONFIDENCE_RE = re.compile(r"\[CONFIDENCE\]\s*([01](?:\.\d+)?)", re.I)
GENERIC_CONFIDENCE_RE = re.compile(
    r"confidence(?:\s+score)?[^0-9]{0,30}([01](?:\.\d+)?)", re.I
)
ITEM_MENTION_RE = re.compile(r"\[?ITEM(?:\s+ID)?\s*[:#-]?\s*(\d+)\]?", re.I)
BRACKET_CONFIDENCE_RE = re.compile(r"\[\s*([01](?:\.\d+)?)\s*\]")
BRACKET_INTEGER_RE = re.compile(r"\[\s*(\d+)\s*\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IRMD CoT candidates from three Qwen2.5-14B temperature teachers")
    parser.add_argument("--model", required=True, help="local Qwen2.5-14B-Instruct path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 0.7, 0.6])
    parser.add_argument("--samples-per-temperature", type=int, default=1)
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=0,
        help="0 uses the default open-catalog prompt; positive values enable a candidate set",
    )
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means every training user")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all-prefixes", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-memory-fraction", type=float, default=0.22)
    parser.add_argument("--max-memory-mib", type=int, default=6200)
    parser.add_argument("--cpu-memory-gib", type=int, default=120)
    parser.add_argument("--offload-folder", default="offload/teacher_14b")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_response(
    text: str, candidates: list[int] | None = None
) -> tuple[str, int | None, float | None]:
    reasoning_match = REASONING_RE.search(text)
    prediction_match = PREDICTION_RE.search(text)
    confidence_match = CONFIDENCE_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    prediction = int(prediction_match.group(1)) if prediction_match else None
    if prediction is None:
        mentions = [int(value) for value in ITEM_MENTION_RE.findall(text)]
        mentions.extend(int(value) for value in BRACKET_INTEGER_RE.findall(text))
        if candidates:
            candidate_set = set(candidates)
            mentions = [value for value in mentions if value in candidate_set]
        prediction = mentions[-1] if mentions else None
    if confidence_match is None:
        matches = GENERIC_CONFIDENCE_RE.findall(text)
        matches.extend(BRACKET_CONFIDENCE_RE.findall(text))
        confidence = float(matches[-1]) if matches else None
    else:
        confidence = float(confidence_match.group(1))
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    return reasoning, prediction, confidence


def chat_text(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    if not path.exists():
        return keys
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            keys.add((record["example_id"], record["teacher_id"], record["sample_index"]))
    return keys


def main() -> None:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("choose only one of --load-in-4bit and --load-in-8bit")
    compute_dtype = torch.bfloat16
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, 0)
        if torch.cuda.get_device_capability(0)[0] < 8:
            compute_dtype = torch.float16
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    done = existing_keys(output)

    corpus = SequenceCorpus.load(args.data_root, args.dataset, args.metadata)
    iterator = corpus.training_examples(all_prefixes=args.all_prefixes)
    examples = list(islice(iterator, args.max_examples or None))
    print(json.dumps({"dataset": corpus.name, **corpus.statistics, "examples": len(examples)}, indent=2))

    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    offload_folder = Path(args.offload_folder)
    offload_folder.mkdir(parents=True, exist_ok=True)
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {0: f"{args.max_memory_mib}MiB", "cpu": f"{args.cpu_memory_gib}GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_folder),
        offload_state_dict=True,
        quantization_config=quantization_config,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    generated = 0
    with output.open("a") as handle, torch.inference_mode():
        for temperature in args.temperatures:
            teacher_id = f"qwen2.5-14b-t{temperature:g}"
            for sample_index in range(args.samples_per_temperature):
                pending = [
                    example
                    for example in examples
                    if (example.example_id, teacher_id, sample_index) not in done
                ]
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start : start + args.batch_size]
                    prompts, candidates_per_example = [], []
                    for example in batch:
                        stable = zlib.crc32(example.example_id.encode())
                        candidates = (
                            corpus.candidates(
                                example,
                                args.candidate_size,
                                args.seed + stable + sample_index,
                            )
                            if args.candidate_size
                            else []
                        )
                        candidates_per_example.append(candidates)
                        prompts.append(
                            teacher_prompt(corpus, example, candidates, args.max_history)
                        )
                    chats = [chat_text(tokenizer, prompt) for prompt in prompts]
                    encoded = tokenizer(
                        chats,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_input_tokens,
                    ).to(model.get_input_embeddings().weight.device)
                    torch.manual_seed(args.seed + start + sample_index * 100003 + int(temperature * 1000))
                    sequences = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                    responses = tokenizer.batch_decode(
                        sequences[:, encoded.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                    for example, prompt, candidates, response in zip(
                        batch, prompts, candidates_per_example, responses
                    ):
                        reasoning, prediction, confidence = parse_response(response, candidates)
                        record = {
                            "example_id": example.example_id,
                            "user_id": example.user_id,
                            "history": example.history,
                            "target": example.target,
                            "candidate_ids": candidates,
                            "teacher_id": teacher_id,
                            "temperature": temperature,
                            "sample_index": sample_index,
                            "teacher_prompt": prompt,
                            "student_prompt": student_prompt(corpus, example, args.max_history),
                            "response": response,
                            "reasoning": reasoning,
                            "prediction": prediction,
                            "confidence": confidence,
                            "prediction_is_candidate": prediction in candidates if candidates and prediction is not None else False,
                            "prediction_is_target": prediction == example.target,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        generated += 1
                    print(
                        json.dumps(
                            {
                                "teacher": teacher_id,
                                "sample": sample_index,
                                "completed": min(start + len(batch), len(pending)),
                                "pending": len(pending),
                                "new_records": generated,
                            }
                        )
                    )
    print(f"wrote {generated} new candidates to {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import zlib
from itertools import islice
from pathlib import Path

from irmd_qwen.data import SequenceCorpus
from irmd_qwen.prompts import (
    SYSTEM_PROMPT,
    grounded_teacher_prompt,
    student_prompt,
    teacher_prompt,
)
from synthesize_teachers import existing_keys, parse_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IRMD teacher candidates with a low-memory Qwen2.5-14B GGUF runtime"
    )
    parser.add_argument("--model", required=True, help="first GGUF shard or a single GGUF file")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 0.7, 0.6])
    parser.add_argument("--samples-per-temperature", type=int, default=1)
    parser.add_argument("--candidate-size", type=int, default=0)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all-prefixes", action="store_true")
    parser.add_argument("--n-threads", type=int, default=24)
    parser.add_argument("--n-batch", type=int, default=256)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--condition-on-target",
        action="store_true",
        help="training-only mode: ask the teacher to explain the observed next item",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    done = existing_keys(output)

    corpus = SequenceCorpus.load(args.data_root, args.dataset, args.metadata)
    iterator = corpus.training_examples(all_prefixes=args.all_prefixes)
    examples = list(islice(iterator, args.max_examples or None))
    print(
        json.dumps(
            {"dataset": corpus.name, **corpus.statistics, "examples": len(examples)},
            indent=2,
        ),
        flush=True,
    )

    from llama_cpp import Llama

    os.environ.setdefault("LLAMA_ARG_NO_WEBUI", "1")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.max_input_tokens + args.max_new_tokens,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        chat_format="chatml",
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )

    generated = 0
    with output.open("a") as handle:
        for temperature in args.temperatures:
            teacher_id = f"qwen2.5-14b-gguf-t{temperature:g}"
            for sample_index in range(args.samples_per_temperature):
                for position, example in enumerate(examples, start=1):
                    key = (example.example_id, teacher_id, sample_index)
                    if key in done:
                        continue
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
                    prompt = (
                        grounded_teacher_prompt(corpus, example, args.max_history)
                        if args.condition_on_target
                        else teacher_prompt(corpus, example, candidates, args.max_history)
                    )
                    result = llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_new_tokens,
                        seed=args.seed + stable + sample_index * 100003 + int(temperature * 1000),
                    )
                    response = result["choices"][0]["message"]["content"] or ""
                    reasoning, prediction, confidence = parse_response(response, candidates)
                    record = {
                        "example_id": example.example_id,
                        "user_id": example.user_id,
                        "history": example.history,
                        "target": example.target,
                        "split": "train",
                        "target_conditioned": args.condition_on_target,
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
                        "prediction_is_candidate": (
                            prediction in candidates
                            if candidates and prediction is not None
                            else False
                        ),
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
                                "completed": position,
                                "examples": len(examples),
                                "new_records": generated,
                            }
                        ),
                        flush=True,
                    )
    print(f"wrote {generated} new candidates to {output}", flush=True)


if __name__ == "__main__":
    main()

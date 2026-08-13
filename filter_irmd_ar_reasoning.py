from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter target-grounded IRMD reasoning candidates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--require-correct", action="store_true")
    parser.add_argument("--max-per-example", type=int, default=1)
    parser.add_argument("--min-reasoning-chars", type=int, default=24)
    parser.add_argument("--diversity-threshold", type=float, default=0.90)
    parser.add_argument(
        "--require-compression-budget",
        action="store_true",
        help="retain only compressed records whose generator marked the word budget valid",
    )
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=1.0,
        help="maximum compressed/source reasoning word ratio; 1 disables the ratio gate",
    )
    return parser.parse_args()


def token_cosine(left: str, right: str) -> float:
    a = Counter(re.findall(r"[a-z0-9]+", left.lower()))
    b = Counter(re.findall(r"[a-z0-9]+", right.lower()))
    if not a or not b:
        return 1.0
    dot = sum(value * b.get(token, 0) for token, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b)


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    rejected = defaultdict(int)
    for record in records:
        if record.get("split") != "train" or not record.get("target_conditioned", False):
            rejected["not_training_target_grounded"] += 1
            continue
        reasoning = (record.get("reasoning") or "").strip()
        confidence = record.get("confidence")
        if len(reasoning) < args.min_reasoning_chars:
            rejected["short_or_missing_reasoning"] += 1
            continue
        if args.require_compression_budget and not record.get(
            "compression_within_word_budget", False
        ):
            rejected["outside_compression_word_budget"] += 1
            continue
        source_words = record.get("source_reasoning_words")
        compressed_words = record.get("reasoning_words")
        if args.max_compression_ratio < 1.0:
            if not source_words or compressed_words is None:
                rejected["missing_compression_lengths"] += 1
                continue
            if float(compressed_words) / float(source_words) > args.max_compression_ratio:
                rejected["insufficient_compression"] += 1
                continue
        if confidence is None:
            rejected["missing_confidence"] += 1
            continue
        if float(confidence) < args.quality_threshold:
            rejected["low_confidence"] += 1
            continue
        if args.require_correct and not record.get("prediction_is_target", False):
            rejected["wrong_prediction"] += 1
            continue
        grouped[record["example_id"]].append(record)

    selected: list[dict] = []
    for example_id in sorted(grouped):
        candidates = sorted(
            grouped[example_id],
            key=lambda row: (float(row["confidence"]), len(row["reasoning"])),
            reverse=True,
        )
        kept: list[dict] = []
        for candidate in candidates:
            if len(kept) >= args.max_per_example:
                break
            if kept and max(
                token_cosine(candidate["reasoning"], previous["reasoning"])
                for previous in kept
            ) >= args.diversity_threshold:
                rejected["low_diversity"] += 1
                continue
            kept.append(candidate)
        selected.extend(kept)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_records": len(records),
        "selected_records": len(selected),
        "selected_examples": len(grouped),
        "selection_rate": len(selected) / max(1, len(records)),
        "quality_threshold": args.quality_threshold,
        "require_correct": args.require_correct,
        "diversity_threshold": args.diversity_threshold,
        "require_compression_budget": args.require_compression_budget,
        "max_compression_ratio": args.max_compression_ratio,
        "rejected": dict(rejected),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not selected:
        raise RuntimeError("reasoning filter retained no records")


if __name__ == "__main__":
    main()

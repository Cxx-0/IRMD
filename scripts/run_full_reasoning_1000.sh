#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-$PWD}
RUNS=${RUNS:-$ROOT/runs}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/full}
METADATA=${METADATA:-$ROOT/data/metadata/Beauty.json}
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
TEACHER_GGUF=${TEACHER_GGUF:?set TEACHER_GGUF to the first Q4_K_M GGUF shard}
PYTHON=${PYTHON:-python}

mkdir -p "$RUNS"

RAW="$RUNS/beauty_teacher3_full_reasoning_300.jsonl"
FILTERED="$RUNS/beauty_teacher3_full_reasoning_300.filtered.jsonl"
BASE="$RUNS/beauty_autoregressive_base"
FULL="$RUNS/beauty_full_reasoning_1000"

# 1. Autoregressive next-item LoRA base on the full training split.
"$PYTHON" pretrain_p5_qwen.py \
  --model "$STUDENT" \
  --data-root "$DATA_ROOT" \
  --dataset Beauty \
  --output "$BASE" \
  --epochs 1 \
  --micro-batch-size 1 \
  --gradient-accumulation 16 \
  --eval-batch-size 1 \
  --eval-max-users 1000 \
  --num-beams 20 \
  --max-input-length 256 \
  --max-new-tokens 8 \
  --lr 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-memory-fraction 0.22 \
  --seed 43

# 2. Three teacher configurations; target conditioning is strictly train-only.
PYTHONPATH=. "$PYTHON" synthesize_irmd_ar_teachers_gguf.py \
  --model "$TEACHER_GGUF" \
  --data-root "$DATA_ROOT" \
  --dataset Beauty \
  --metadata "$METADATA" \
  --output "$RAW" \
  --temperatures 0.3 0.7 1.0 \
  --samples-per-temperature 1 \
  --max-history 20 \
  --max-examples 300 \
  --max-input-tokens 3072 \
  --max-new-tokens 256 \
  --seed 2026 \
  --n-threads 6 \
  --n-batch 256 \
  --n-gpu-layers 0 \
  --condition-on-target

# 3. Dual constraints plus automatic target-correctness validation.
"$PYTHON" filter_irmd_ar_reasoning.py \
  --input "$RAW" \
  --output "$FILTERED" \
  --quality-threshold 0.85 \
  --require-correct \
  --max-per-example 3 \
  --diversity-threshold 0.90

# 4. Continue from the autoregressive base with full explicit reasoning.
"$PYTHON" train_irmd_ar.py \
  --model "$STUDENT" \
  --adapter "$BASE" \
  --data-root "$DATA_ROOT" \
  --dataset Beauty \
  --metadata "$METADATA" \
  --corpus "$FILTERED" \
  --output "$FULL" \
  --epochs 1 \
  --micro-batch-size 1 \
  --gradient-accumulation 16 \
  --eval-batch-size 1 \
  --eval-max-users 1000 \
  --num-beams 20 \
  --max-input-length 256 \
  --max-new-tokens 8 \
  --max-train-users 1000 \
  --use-all-training-users \
  --lr 8e-7 \
  --recommendation-weight 1.0 \
  --reasoning-weight 0.35 \
  --implicit-weight 0.0 \
  --reasoning-fraction 1.0 \
  --max-memory-fraction 0.22 \
  --seed 43

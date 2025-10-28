#!/usr/bin/env bash
set -e

OUT_DIR="results/cora/nudity"
CSV_OUT="per_row_nudenet.csv"

mkdir -p "$(dirname "$CSV_OUT")"

python metrics/src/eval_nudenet.py \
  --results_dir "$OUT_DIR" \
  --threshold 0.3 \
  --save_csv "$CSV_OUT"

#!/bin/bash
set -Eeuo pipefail

mkdir -p metrics/results

# ---------- Job 1: NEUTRAL ----------
echo "[`date`] Launching NEUTRAL generalization run..."
CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_generalization.py \
  --original_dir 'outputs_adavd/Donald Trump/original' \
  --edited_dir   'outputs_adavd/Donald Trump/retain' \
  --target "Donald Trump" \
  --aliases "President of the United States of America" \
  --replacement "" \
  --r_report \
  --unrelated "Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  --e_thresh 0.27 \
  --out_csv metrics/results/per_image_generalization.csv \
  > metrics/results/adavd_generalization_trump.log 2>&1 &

PID1=$!
# Wait for Job 1 to finish
wait "$PID1"
echo "[`date`] Run finished."
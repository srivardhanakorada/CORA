#!/bin/bash
set -Eeuo pipefail

mkdir -p metrics/results

# ---------- Job 1: NEUTRAL ----------
echo "[`date`] Launching NEUTRAL generalization run..."
CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_generalization.py \
  --original_dir 'outputs_cora_anc/Donald Trump/neut/original' \
  --edited_dir   'outputs_cora_anc/Donald Trump/neut/erase' \
  --target "Donald Trump" \
  --aliases "President of the United States of America" \
  --replacement "a celebrity" \
  --r_report \
  --unrelated "Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  --e_thresh 0.24 \
  --out_csv metrics/results/per_image_generalization_neut.csv \
  > metrics/results/cora_generalization_trump_neut.log 2>&1 &

PID1=$!
echo "[`date`] NEUTRAL PID: $PID1 (logs: metrics/results/cora_generalization_trump_neut.log)"

# Wait for Job 1 to finish
wait "$PID1"
echo "[`date`] NEUTRAL run finished."

# ---------- Job 2: INTENDED ----------
echo "[`date`] Launching INTENDED generalization run..."
CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_generalization.py \
  --original_dir 'outputs_cora_anc/Donald Trump/int/original' \
  --edited_dir   'outputs_cora_anc/Donald Trump/int/erase' \
  --target "Donald Trump" \
  --aliases "President of the United States of America" \
  --replacement "Tom Cruise" \
  --r_report \
  --unrelated "Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  --e_thresh 0.24 \
  --out_csv metrics/results/per_image_generalization_int.csv \
  > metrics/results/cora_generalization_trump_int.log 2>&1 &

PID2=$!
echo "[`date`] INTENDED PID: $PID2 (logs: metrics/results/cora_generalization_trump_int.log)"

# (Optional) Wait for Job 2 as well; comment out if you want the script to exit immediately.
wait "$PID2"
echo "[`date`] INTENDED run finished. All done."
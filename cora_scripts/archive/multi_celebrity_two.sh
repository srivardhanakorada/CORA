#!/usr/bin/env bash
set -e

OUT_DIR="results/cora/multi/two"
LOGFILE="logs/cora_multi_two.log"

PYTHON="python"
SCRIPT="cora_src/main_cora_multi.py"
CSV="cora_src/data/two.csv"

mkdir -p "$(dirname "$LOGFILE")" "$OUT_DIR"

CUDA_VISIBLE_DEVICES=0 nohup \
  "$PYTHON" "$SCRIPT" \
    --save_path "$OUT_DIR" \
    --mode "original,erase" \
    --erase_type celebs \
    --concepts_csv "$CSV" \
    --single_anchor_mode \
    --single_anchor_text "a man" \
    --beta 0.5 --tau 0.1 \
    --num_samples 10 --batch_size 10 --total_timesteps 30 \
    --guidance_scale 7.5 \
    --xformers --tf32 --debug \
  >"$LOGFILE" 2>&1 < /dev/null &

echo "PID: $!"
echo "Log: $LOGFILE"

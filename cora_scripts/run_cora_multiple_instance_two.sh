#!/usr/bin/env bash
set -e

OUT_DIR="outputs_cora_multi_small/instance/two"
LOGFILE="logs/cora_multi_instance_small_two.log"

PYTHON="python"
SCRIPT="cora_src/main_cora_multi.py"
CSV="cora_src/instance_ten_two.csv"

mkdir -p "$(dirname "$LOGFILE")" "$OUT_DIR"

CUDA_VISIBLE_DEVICES=1 nohup \
  "$PYTHON" "$SCRIPT" \
    --save_path "$OUT_DIR" \
    --mode "original,erase" \
    --erase_type simple \
    --concepts_csv "$CSV" \
    --single_anchor_mode \
    --single_anchor_text "an Orange" \
    --beta 0.5 --tau 0.1 \
    --num_samples 100 --batch_size 10 --total_timesteps 30 \
    --guidance_scale 7.5 \
    --xformers --tf32 --debug \
  >"$LOGFILE" 2>&1 < /dev/null &

echo "PID: $!"
echo "Log: $LOGFILE"

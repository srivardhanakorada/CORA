#!/usr/bin/env bash
set -e

# Change GPU by prefixing: GPU_ID=1 ./run_cora.sh
GPU_ID="${GPU_ID:-2}"

OUT_DIR="outputs_cora_multi/fifty"
LOGFILE="logs/cora_multi_celebrity_fifty.log"

PYTHON="python"
SCRIPT="cora_src/main_cora_multi.py"
CSV="cora_src/celebrity_fifty.csv"

mkdir -p "$(dirname "$LOGFILE")" "$OUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup \
  "$PYTHON" "$SCRIPT" \
    --save_path "$OUT_DIR" \
    --mode "original,erase" \
    --erase_type simple \
    --concepts_csv "$CSV" \
    --single_anchor_mode \
    --single_anchor_text "a person" \
    --beta 0.5 --tau 0.1 \
    --num_samples 10 --batch_size 10 --total_timesteps 30 \
    --guidance_scale 7.5 \
    --xformers --tf32 --debug \
  >"$LOGFILE" 2>&1 < /dev/null &

echo "PID: $!"
echo "Log: $LOGFILE"

#!/usr/bin/env bash
set -e

OUT_DIR="results/cora/nudity"
LOGFILE="logs/cora_nudity.log"

PYTHON="python"
SCRIPT="cora_src/main_cora_nudity.py"
CSV="data/i2p_benchmark.csv"

mkdir -p "$(dirname "$LOGFILE")" "$OUT_DIR"

# Notes:
# - If your CSV uses bare 'stable-diffusion-v1-4', the script auto-maps to 'CompVis/stable-diffusion-v1-4'.
# - Add --local_only to forbid Hub downloads and use only local cache/paths.

CUDA_VISIBLE_DEVICES=0 nohup \
  "$PYTHON" "$SCRIPT" \
    --save_path "$OUT_DIR" \
    --mode "original,erase" \
    --erase_type nsfw \
    --concepts_csv "$CSV" \
    --target_concept "nudity" \
    --single_anchor_mode \
    --single_anchor_text "a man" \
    --preserve_concepts "bikini, swimsuit, portrait, beautiful, glamorous" \
    --beta 0.5 --tau 0.1 \
    --num_samples 4709 --batch_size 10 --total_timesteps 30 \
    --guidance_scale 7.5 \
    --sd_ckpt "CompVis/stable-diffusion-v1-4" \
    --xformers --tf32 --debug \
  >"$LOGFILE" 2>&1 < /dev/null &

echo "PID: $!"
echo "Log: $LOGFILE"

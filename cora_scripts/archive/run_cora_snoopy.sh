#!/bin/bash
LOGFILE="logs/cora_snoopy.log"
TARGET="Snoopy"
CONTENTS="Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator"
ROOT_PATH="outputs_cora/$TARGET"
PRETRAIN_PATH="data/pretrain/instance"

echo "Launching CORA for target: $TARGET"
CUDA_VISIBLE_DEVICES=0 nohup python src/main_cora_three.py \
  --save_root outputs_cora \
  --mode "original,erase" \
  --erase_type instance \
  --target_concept "$TARGET" \
  --anchor_concept "Dog" \
  --preserve_concepts "Mickey, Spongebob, Pikachu" \
  --contents "$CONTENTS" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 5 --total_timesteps 30 --guidance_scale 7.5 --xformers --tf32 \
  > "$LOGFILE" 2>&1 < /dev/null &

PID=$!
echo "CORA launched in background with PID $PID. Logs: $LOGFILE"

wait $PID
echo "=== CORA generation finished for $TARGET ==="

CUDA_VISIBLE_DEVICES=0 python src/clip_score_cal.py \
  --contents "$CONTENTS" \
  --root_path "$ROOT_PATH" \
  --pretrained_path "$PRETRAIN_PATH"

echo "=== Evaluation complete. Results written to $ROOT_PATH/record_metrics.txt ==="
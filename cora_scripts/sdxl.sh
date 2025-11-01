#!/bin/bash

# Define output log file
LOGFILE="logs/sdxl.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_sdxl.py \
  --prompt "A photo of a crow" \
  --target_concept "Crow" \
  --anchor_concept "Yellow Parrot" \
  --preserve_concepts "Eagle, Pigeon" \
  --seed 0 \
  --num_samples 10 --batch_size 2 --steps 30 \
  --guidance_scale 7.5 \
  --xformers \
  --save_path "results/cora/sdxl/crow" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
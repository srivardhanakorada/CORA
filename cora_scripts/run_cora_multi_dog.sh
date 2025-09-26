#!/bin/bash

# Define output log file
LOGFILE="logs/cora_multi_dog.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python cora_src/main_cora_multi.py \
  --save_root outputs_cora_multi \
  --mode "original,erase" \
  --erase_type simple \
  --target_concepts "Dog,Crow" \
  --anchor_pools "cat|pigeon" \
  --preserve_concepts "Tiger,Eagle,Parrot" \
  --contents "Dog,Crow,Tiger,Eagle,Parrot,Cat,Dolphin" \
  --num_samples 100 --batch_size 10 \
  --total_timesteps 30 --guidance_scale 7.5 \
  --beta 0.5 --tau 0.1 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
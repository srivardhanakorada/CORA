#!/bin/bash

# Define output log file
LOGFILE="logs/cora_anc_snoopy.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --save_root outputs_cora_anc \
  --mode "original,erase" \
  --erase_type instance \
  --target_concept "Snoopy" \
  --anchor_pool "dog,cartoon-dog,scooby-doo" \
  --preserve_concepts "Pluto,Donald Duck,Goofy" \
  --contents "Snoopy,Mickey,Spongebob,Pikachu,Dog,Legislator" \
  --num_samples 10 --batch_size 10 \
  --total_timesteps 30 --guidance_scale 7.5 \
  --beta 0.5 --tau 0.1 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
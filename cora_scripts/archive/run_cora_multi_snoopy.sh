#!/bin/bash

# Define output log file
LOGFILE="logs/cora_multi_snoopy.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_multi.py \
  --save_root outputs_cora_multi \
  --mode "original,erase" \
  --erase_type instance \
  --target_concepts "Snoopy,Mickey,Spongebob" \
  --anchor_pools "dog,cartoon-dog,scooby-doo|mouse,rat,cartoon-mouse|cheese,cartoon-cheese" \
  --preserve_concepts "Pluto,Sponge,Donald Duck,Mouse,Goofy" \
  --contents "Snoopy,Mickey,Spongebob,Pikachu,Dog,Legislator" \
  --num_samples 10 --batch_size 10 \
  --total_timesteps 30 --guidance_scale 7.5 \
  --beta 0.5 --tau 0.1 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
#!/bin/bash

# Define output log file
LOGFILE="cora_multi.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_multi.py \
  --save_root outputs_cora_multi \
  --mode "original,erase" \
  --erase_type simple \
  --target_concepts "Donald Trump,Mickey Mouse" \
  --anchor_pools "a man,a celebrity,a politician|a mouse,a rat,an animal" \
  --preserve_concepts "Elon Musk, Barack Obama, Donald Duck, Goofy" \
  --contents "Donald Trump,Mickey Mouse,Elon Musk, Barack Obama, Donald Duck, Goofy, President of United States of America, Bill Gates, Sachin Tendulkar, Mouse, Rat" \
  --num_samples 10 --batch_size 10 \
  --total_timesteps 30 --guidance_scale 7.5 \
  --beta 0.5 --tau 0.1 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
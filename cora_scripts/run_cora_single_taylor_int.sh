#!/bin/bash

# Define output log file
LOGFILE="logs/cora_single_taylor_int.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python cora_src/main_cora_anc.py \
  --save_root outputs_cora_anc \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Taylor Swift" \
  --anchor_pool "Selena Gomez" \
  --preserve_concepts "Ed Sheeran, Ariana Grande" \
  --contents "Taylor Swift, Selena Gomez, Ed Sheeran, Ariana Grande, Sachin Tendulkar, Anne Hathaway, Bill Gates, President of United States of America" \
  --beta 0.5 --tau 0.1 \
  --num_samples 1000 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "outputs_cora_anc/Taylor Swift/int" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
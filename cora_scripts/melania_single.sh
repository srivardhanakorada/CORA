#!/bin/bash

# Define output log file
LOGFILE="logs/melania_single_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type celebs \
  --target_concept "Melania Trump" \
  --anchor_pool "a woman" \
  --preserve_concepts "Donald Trump, Michelle Obama" \
  --contents "Bruce Lee, Marilyn Monroe, Anne Hathaway, Tom Cruise, Wife of Donald Trump, First Lady of United States of America" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/melania" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
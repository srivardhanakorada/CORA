#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_snoopy_mickey_spongebob.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python adavd_src/main_multi.py \
  --save_root outputs_adavd \
  --mode "original,retain" \
  --erase_type 'advd_mul_sim' \
  --contents 'erase, retention' \
  --target_concept "Snoopy,Mickey,Spongebob,Dog,Legislator,Pikachu" \
  --num_samples 10 --batch_size 10 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
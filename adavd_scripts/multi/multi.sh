#!/bin/bash

# Define output log file
LOGFILE='logs/adavd_multi.log'

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_multi.py \
  --save_root results/adavd \
  --mode original,retain \
  --erase_type 'dup' \
  --contents 'erase, retention' \
  --target_concept 'Donald Trump,Dwayne Johnson' \
  --num_samples 100 --batch_size 10 \
  > $LOGFILE 2>&1 < /dev/null &
echo PID: $!
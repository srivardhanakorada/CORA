#!/bin/bash

# Define output log file
LOGFILE='logs/adavd_multi_two.log'

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_multi.py \
  --save_root results/adavd/multi/two \
  --mode original,retain \
  --erase_type 'celebs_two' \
  --contents 'erase, retention' \
  --target_concept 'Adam Driver, Adriana Lima' \
  --num_samples 10 --batch_size 10 \
  > $LOGFILE 2>&1 < /dev/null &
echo PID: $!
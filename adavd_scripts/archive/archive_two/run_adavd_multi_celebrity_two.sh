#!/bin/bash

# Define output log file
LOGFILE='logs/adavd_multi_celebrity_two.log'

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_multi.py \
  --save_root outputs_adavd_multi_small \
  --mode original,retain \
  --erase_type 'celebrity_two' \
  --contents 'erase, retention' \
  --target_concept 'Adam Driver, Adriana Lima' \
  --num_samples 100 --batch_size 10 \
  > $LOGFILE 2>&1 < /dev/null &
echo PID: $!
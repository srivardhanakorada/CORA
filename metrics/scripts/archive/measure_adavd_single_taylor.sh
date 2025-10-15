#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_adavd/Taylor Swift/original' \
  --edited_intended_dir 'outputs_adavd/Taylor Swift/retain' \
  --edited_neutral_dir 'outputs_adavd/Taylor Swift/retain' \
  --target "Taylor Swift" \
  --replacement "" \
  --neutral "" \
  --unrelated "Ed Sheeran, Ariana Grande, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  > metrics/results/adavd_single_taylor.log 2>&1 &
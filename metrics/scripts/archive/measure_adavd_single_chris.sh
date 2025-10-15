#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_adavd/Chris Evans/original' \
  --edited_intended_dir 'outputs_adavd/Chris Evans/retain' \
  --edited_neutral_dir 'outputs_adavd/Chris Evans/retain' \
  --target "Chris Evans" \
  --replacement "" \
  --neutral "" \
  --unrelated "Sebastian Stan, Tom Hiddleston, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  > metrics/results/adavd_single_chris.log 2>&1 &
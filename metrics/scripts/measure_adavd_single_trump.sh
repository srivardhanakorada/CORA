#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_adavd/Donald Trump/original' \
  --edited_intended_dir 'outputs_adavd/Donald Trump/retain' \
  --edited_neutral_dir 'outputs_adavd/Donald Trump/retain' \
  --target "Donald Trump" \
  --replacement "" \
  --neutral "" \
  --unrelated "Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  > metrics/results/adavd_single_trump.log 2>&1 &
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_cora_anc/Donald Trump/neut/original' \
  --edited_intended_dir 'outputs_cora_anc/Donald Trump/int/erase' \
  --edited_neutral_dir 'outputs_cora_anc/Donald Trump/neut/erase' \
  --target "Donald Trump" \
  --replacement "Tom Cruise" \
  --neutral "a celebrity" \
  --unrelated "Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  > metrics/results/cora_single_trump.log 2>&1 &
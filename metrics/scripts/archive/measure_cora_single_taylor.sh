#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_cora_anc/Taylor Swift/neut/original' \
  --edited_intended_dir 'outputs_cora_anc/Taylor Swift/int/erase' \
  --edited_neutral_dir 'outputs_cora_anc/Taylor Swift/neut/erase' \
  --target "Taylor Swift" \
  --replacement "Selena Gomez" \
  --neutral "a singer" \
  --unrelated "Ed Sheeran, Ariana Grande, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  > metrics/results/cora_single_taylor.log 2>&1 &
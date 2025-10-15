#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_cora_anc/Apple/neut/original' \
  --edited_intended_dir 'outputs_cora_anc/Apple/int/erase' \
  --edited_neutral_dir 'outputs_cora_anc/Apple/neut/erase' \
  --target "an Apple" \
  --replacement "an Orange" \
  --neutral "a food item" \
  --unrelated "a Pineapple, a Custard apple, a Banana, a Jackfruit, a Lemon" \
  > metrics/results/cora_single_apple.log 2>&1 &
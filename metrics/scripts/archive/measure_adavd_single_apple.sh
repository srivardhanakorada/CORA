#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_single.py \
  --original_dir 'outputs_adavd/Apple/original' \
  --edited_intended_dir 'outputs_adavd/Apple/retain' \
  --edited_neutral_dir 'outputs_adavd/Apple/retain' \
  --target "an Apple" \
  --replacement "" \
  --neutral "" \
  --unrelated "a Pineapple, a Custard apple, a Banana, a Jackfruit, a Lemon" \
  > metrics/results/adavd_single_apple.log 2>&1 &
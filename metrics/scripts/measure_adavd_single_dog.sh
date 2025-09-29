#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_single.py \
  --original_dir 'outputs_adavd/Dog/original' \
  --edited_intended_dir 'outputs_adavd/Dog/retain' \
  --edited_neutral_dir 'outputs_adavd/Dog/retain' \
  --target "Dog" \
  --replacement "" \
  --neutral "" \
  --unrelated "Lion, Wolf, Rat, Cow, Goat" \
  > metrics/results/adavd_single_dog.log 2>&1 &
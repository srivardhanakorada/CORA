#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_single.py \
  --original_dir 'outputs_adavd/Crow/original' \
  --edited_intended_dir 'outputs_adavd/Crow/retain' \
  --edited_neutral_dir 'outputs_adavd/Crow/retain' \
  --target "Crow" \
  --replacement "" \
  --neutral "" \
  --unrelated "Raven, Vulture, Parrot, Sparrow" \
  > metrics/results/adavd_single_crow.log 2>&1 &
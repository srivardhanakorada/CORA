#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_cora_anc/Crow/neut/original' \
  --edited_intended_dir 'outputs_cora_anc/Crow/int/erase' \
  --edited_neutral_dir 'outputs_cora_anc/Crow/neut/erase' \
  --target "Crow" \
  --replacement "Eagle" \
  --neutral "an animal" \
  --unrelated "Raven, Vulture, Parrot, Sparrow" \
  > metrics/results/cora_single_crow.log 2>&1 &
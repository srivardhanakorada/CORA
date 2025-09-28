#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_cora_anc/Dog/neut/original' \
  --edited_intended_dir 'outputs_cora_anc/Dog/int/erase' \
  --edited_neutral_dir 'outputs_cora_anc/Dog/neut/erase' \
  --target "Dog" \
  --replacement "Cat" \
  --neutral "a pet" \
  --unrelated "Lion, Wolf, Rat, Cow, Goat" \
  > metrics/results/cora_single_dog.log 2>&1 &
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/trial_one.py \
  --original_dir 'outputs_cora_anc/Chris Evans/neut/original' \
  --edited_intended_dir 'outputs_cora_anc/Chris Evans/int/erase' \
  --edited_neutral_dir 'outputs_cora_anc/Chris Evans/neut/erase' \
  --target "Chris Evans" \
  --replacement "Chris Hemsworth" \
  --neutral "an actor" \
  --unrelated "Sebastian Stan, Tom Hiddleston, Sachin Tendulkar, Anne Hathaway, Bill Gates" \
  > metrics/results/cora_single_chris.log 2>&1 &
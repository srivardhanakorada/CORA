#!/bin/bash
set -Eeuo pipefail

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_distance.py \
  --cora_root outputs_cora_anc \
  --adavd_root outputs_adavd \
  --out_csv metrics/results/fid_summary.csv \
  --batch_size 64 --num_workers 4 --device cuda:0 > metrics/results/calc_dis.log 2>&1 &
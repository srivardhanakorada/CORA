#!/bin/bash
set -Eeuo pipefail

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_distance_multiple.py \
  --src_root outputs_cora_multi_small/celebrity/one/original \
  --tgt_root outputs_cora_multi_small/celebrity/one/erase \
  --concepts 'Adam Driver'\
  --out_csv metrics/fid_summary_one.csv > metrics/results/calc_dis_one.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -W ignore metrics/measure_distance_multiple.py \
  --src_root outputs_cora_multi_small/celebrity/two/original \
  --tgt_root outputs_cora_multi_small/celebrity/two/erase \
  --concepts 'Adam Driver,Adriana Lima'\
  --out_csv metrics/fid_summary_two.csv > metrics/results/calc_dis_two.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -W ignore metrics/measure_distance_multiple.py \
  --src_root outputs_cora_multi_small/celebrity/five/original \
  --tgt_root outputs_cora_multi_small/celebrity/five/erase \
  --concepts 'Adam Driver,Adriana Lima,Amber Heard,Amy Adams,Andrew Garfield'\
  --out_csv metrics/fid_summary_five.csv > metrics/results/calc_dis_five.log 2>&1 &
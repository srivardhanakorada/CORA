#!/bin/bash
set -Eeuo pipefail

# CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_distance_multiple.py \
#   --src_root outputs_adavd_multi_small/celebrity/one/original \
#   --tgt_root outputs_adavd_multi_small/celebrity/one/erase \
#   --concepts 'Adam Driver,Adriana Lima,Amber Heard,Amy Adams,Andrew Garfield,Angelina Jolie,Anjelica Huston,Anna Faris,Anna Kendrick,Anne Hathaway'\
#   > metrics/results/calc_dis_one.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -W ignore metrics/measure_distance_multiple.py \
#   --src_root outputs_adavd_multi_small/celebrity/two/original \
#   --tgt_root outputs_adavd_multi_small/celebrity/two/erase \
#   --concepts 'Adam Driver,Adriana Lima,Amber Heard,Amy Adams,Andrew Garfield,Angelina Jolie,Anjelica Huston,Anna Faris,Anna Kendrick,Anne Hathaway'\
#   > metrics/results/calc_dis_two.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -W ignore metrics/measure_distance_multiple.py \
  --src_root outputs_adavd_multi_small/celebrity/five/original \
  --tgt_root outputs_adavd_multi_small/celebrity/five/erase \
  --concepts 'Adam Driver,Adriana Lima,Amber Heard,Amy Adams,Andrew Garfield,Angelina Jolie,Anjelica Huston,Anna Faris,Anna Kendrick,Anne Hathaway'\
  > metrics/results/calc_dis_five.log 2>&1 &
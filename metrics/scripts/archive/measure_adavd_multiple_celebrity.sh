#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore metrics/measure_multiple.py \
  --original_dir outputs_adavd_multi_small/celebrity/one/original \
  --neutral_dir outputs_adavd_multi_small/celebrity/one/erase \
  --targets "Adam Driver" \
  --unrelated "Adriana Lima,Amber Heard,Amy Adams,Andrew Garfield,Angelina Jolie,Anjelica Huston,Anna Faris,Anna Kendrick,Anne Hathaway" \
  > metrics/results/adavd_multiple_celebrities_one.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -W ignore metrics/measure_multiple.py \
  --original_dir outputs_adavd_multi_small/celebrity/two/original \
  --neutral_dir outputs_adavd_multi_small/celebrity/two/erase \
  --targets "Adam Driver,Adriana Lima" \
  --unrelated "Amber Heard,Amy Adams,Andrew Garfield,Angelina Jolie,Anjelica Huston,Anna Faris,Anna Kendrick,Anne Hathaway"\
  > metrics/results/adavd_multiple_celebrities_two.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -W ignore metrics/measure_multiple.py \
  --original_dir outputs_adavd_multi_small/celebrity/five/original \
  --neutral_dir outputs_adavd_multi_small/celebrity/five/erase \
  --targets "Adam Driver,Adriana Lima,Amber Heard,Amy Adams,Andrew Garfield" \
  --unrelated "Angelina Jolie,Anjelica Huston,Anna Faris,Anna Kendrick,Anne Hathaway"\
  > metrics/results/adavd_multiple_celebrities_five.log 2>&1 &
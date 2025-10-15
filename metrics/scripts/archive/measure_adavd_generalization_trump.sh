#!/bin/bash
set -Eeuo pipefail

## thresholding
# nohup python metrics/measure_threshold_for_generalization.py \
#       --folder "outputs_cora_single/Donald Trump/neut/original" \
#       --targets "Donald Trump" \
#       --non_targets "Barack Obama, Elon Musk" \
#       --text "Donald Trump" \
#       --recall 0.95 \
#       --save_csv results/clip_trump_scores.csv > results/log_trump.log 2>&1 &

# ## actual measure
# nohup python -W ignore metrics/measure_generalization.py \
#       --folder "outputs_adavd_single/Donald Trump/original" \
#       --concept "President of the United States of America" \
#       --threshold 0.2110

## actual measure
nohup python -W ignore metrics/measure_generalization.py \
      --folder "outputs_adavd_single/Donald Trump/retain" \
      --concept "President of the United States of America" \
      --threshold 0.2110
#!/bin/bash
set -Eeuo pipefail

# ## thresholding
# nohup python metrics/measure_threshold_for_generalization.py \
#       --folder "outputs_cora_single/Donald Trump/neut/original" \
#       --targets "Donald Trump" \
#       --non_targets "Barack Obama, Elon Musk" \
#       --text "Donald Trump" \
#       --recall 0.95 \
#       --save_csv results/clip_trump_scores.csv > results/log_trump.log 2>&1 &

# ## actual measure
# nohup python -W ignore metrics/measure_generalization.py \
#       --folder "outputs_cora_single/Donald Trump/neut/original" \
#       --concept "President of the United States of America" \
#       --threshold 0.2110

# ## actual measure
# nohup python -W ignore metrics/measure_generalization.py \
#       --folder "outputs_cora_single/Donald Trump/neut/erase" \
#       --concept "President of the United States of America" \
#       --threshold 0.2110

# ## thresholding
# nohup python metrics/measure_threshold_for_generalization.py \
#       --folder "outputs_cora_single/Dog/neut/original" \
#       --targets "Dog" \
#       --non_targets "Cat, Lion" \
#       --text "Dog" \
#       --recall 0.95 \
#       --save_csv results/clip_trump_dog.csv > results/log_dog.log 2>&1 &

## actual measure
# nohup python -W ignore metrics/measure_generalization.py \
#       --folder "outputs_cora_anc/Dog/gen/original" \
#       --concept "mans best friend" \
#       --threshold 0.1908

# ## actual measure
# nohup python -W ignore metrics/measure_generalization.py \
#       --folder "outputs_cora_anc/Dog/gen/erase" \
#       --concept "mans best friend" \
#       --threshold 0.1908

## thresholding
# nohup python metrics/measure_threshold_for_generalization.py \
#       --folder "outputs_cora_anc/Spider/gen/original" \
#       --targets "Spider Man" \
#       --non_targets "Venom" \
#       --text "Spider Man" \
#       --recall 0.95 \
#       --save_csv results/clip_trump_spider.csv > results/log_spider.log 2>&1 &

## actual measure
# nohup python -W ignore metrics/measure_generalization.py \
#       --folder "outputs_cora_anc/Spider/gen/original" \
#       --concept "Marvel superhero who got bitten by a radio active spider wearing red and blue suit" \
#       --threshold 0.2293

# ## actual measure
nohup python -W ignore metrics/measure_generalization.py \
      --folder "outputs_cora_anc/Spider/gen/erase" \
      --concept "Marvel superhero who got bitten by a radio active spider wearing red and blue suit" \
      --threshold 0.2293
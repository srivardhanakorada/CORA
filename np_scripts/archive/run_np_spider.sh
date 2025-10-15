#!/bin/bash

# Define output log file
LOGFILE="np_spider.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python src/negative_prompting/main.py \
  --out_root "outputs_np" \
  --target_root "Spider Man" \
  --names "Spider Man" "Iron Man" "Venom" "Bat Man" "Super Man" "Marvel superhero who got bitten by a radio active spider wearing red and blue suit" \
  --negative_prompt "Spider Man" \
  --num_samples 1000 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
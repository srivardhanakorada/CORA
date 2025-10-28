#!/bin/bash

# Define output log file
LOGFILE="logs/multi_np.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python np_src/main.py \
  --out_root "results/np/multi" \
  --names "Donald Trump" "Dwayne Johnson" "Mark Zuckerberg" "Prince William" \
    "Robert Downey" "President of the United States of America" \
    "Husband of Melania Trump" "First Lady of United States" \
    "Wife of Donald Trump" "CEO of Facebook" \
    "Founder of Facebook" "Husband of Kate Middleton" \
    "Future King of England" "Actor who played Iron Man" "Tony Stark" \
    "Elon Musk" "Tom Cruise" \
  --negative_prompt "Donald Trump, Dwayne Johnson, Mark Zuckerberg, Prince William, Robert Downey" \
  --num_samples 100 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
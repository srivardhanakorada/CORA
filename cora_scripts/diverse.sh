#!/bin/bash

# Define output log file
LOGFILE="logs/diverse.log"

# Run the CoRA script with nohup
# CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_anc.py \
#   --mode "original,erase" \
#   --erase_type celebs \
#   --target_concept "Snow" \
#   --anchor_pool "Sand" \
#   --preserve_concepts "Water, Rivers, Mountains" \
#   --contents "Snow Mountain, Lake covered in Snow" \
#   --beta 0.5 --tau 0.1 \
#   --num_samples 10 --batch_size 10 --total_timesteps 30 \
#   --guidance_scale 7.5 \
#   --xformers --tf32 \
#   --debug \
#   --save_path "results/cora/diverse/snow" \
#   > "$LOGFILE" 2>&1 < /dev/null &
# echo "PID: $!"

CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type celebs \
  --target_concept "Fog" \
  --anchor_pool "Rain" \
  --preserve_concepts "Snow" \
  --contents "Fog covered road, Fog lifting up in early morning" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/diverse/fog" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
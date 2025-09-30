CUDA_VISIBLE_DEVICES=0 python -W ignore metrics/measure_distance_target.py \
  --cora_root outputs_cora_anc \
  --adavd_root outputs_adavd \
  --out_csv metrics/results/fid_concept_summary.csv \
  --whitelist_file metrics/concepts.txt \
  --device cuda:0

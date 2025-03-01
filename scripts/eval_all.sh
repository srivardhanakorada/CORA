#!/bin/bash

declare -A targets_map
declare -A contents_map

# ========== 输入参数组 ==========
# Erase Mode Config
mode="retain"

# Erase Task Config
# erase_types=("style")
# targets_map["style"]="Monet"
erase_types=("instance" "style" "celebrity")
targets_map["instance"]="Snoopy;Snoopy, Mickey;Snoopy, Mickey, Spongebob"
targets_map["style"]="Van Gogh;Picasso;Monet"
targets_map["celebrity"]="Bruce Lee;Marilyn Monroe;Melania Trump"

# Hyper Config
param_groups=(
  "100 0.93 2.0"
)

# GPU Config
GPU_IDX=('0')
# ============================

contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator"
contents_map["style"]="Van Gogh, Picasso, Monet, Andy Warhol, Caravaggio"
contents_map["celebrity"]="Bruce Lee, Marilyn Monroe, Melania Trump, Anne Hathaway, Tom Cruise"

# 定义要使用的 GPU 索引数组
NUM_GPUS=${#GPU_IDX[@]}  # 计算 GPU 数量
gpu_idx=0 # 初始化 GPU 分配的索引

# 函数：提交任务到指定 GPU
run_task() {
  local erase_type=$1
  local target=$2
  local content=$3
  local gpu_id=$4
  local a=$5
  local b=$6
  local c=$7
  local save_root=$8

  echo "Running task for $erase_type with $target on GPU $gpu_id with a=$a, b=$b, c=$c"

  CUDA_VISIBLE_DEVICES=$gpu_id python src/main.py \
    --erase_type "$erase_type" \
    --target_concept "$target" \
    --contents "$content" \
    --mode "$mode" \
    --num_samples 10 --batch_size 10 \
    --sigmoid_a "$a" --sigmoid_b "$b" --sigmoid_c "$c" \
    --save_root "$save_root" 
}

# 遍历所有参数组
for params in "${param_groups[@]}"; do
  # 提取参数 a, b, c
  read a b c <<< "$params"
  save_root="logs/${a}_${b}_${c}"

  # 遍历所有任务
  for erase_type in "${erase_types[@]}"; do

    IFS=';' read -ra targets <<< "${targets_map[$erase_type]}"
    IFS=',' read -ra contents <<< "${contents_map[$erase_type]}"

    for target in "${targets[@]}"; do

      for content in "${contents[@]}"; do

        # 对指定 target 和 content 在单 GPU 上采样
        run_task "$erase_type" "$target" "$content" "${GPU_IDX[$gpu_idx]}" "$a" "$b" "$c" "${save_root}/${erase_type}" &

        # 更新 GPU 索引，循环使用 GPU
        gpu_idx=$((gpu_idx + 1))

        # 检查是否超过 NUM_GPUS，如果超过则 wait
        if (( gpu_idx >= NUM_GPUS )); then
            wait
            gpu_idx=0  # 重置 GPU 索引
        fi

      done

      wait # 等待该 target 对应的所有 contents 跑完，进行 evaluation
      gpu_idx=0  # 重置 GPU 索引

      CUDA_VISIBLE_DEVICES=${GPU_IDX[0]} python src/clip_score_cal.py \
          --contents "${contents_map[$erase_type]}" \
          --root_path "${save_root}/${erase_type}/$(echo "$target" | sed 's/, /_/g')" \
          --pretrained_path "data/pretrain/${erase_type}"

    done
  done
done

# 等待最后一批任务完成
wait
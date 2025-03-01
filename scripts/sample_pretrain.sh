#!/bin/bash

# 关联数组：任务参数
declare -A target_concept_map
declare -A contents_map

# 定义目标概念和内容
target_concept_map["instance"]="instance"
target_concept_map["style"]="style"
target_concept_map["celebrity"]="celebrity"

contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator"
contents_map["style"]="Van Gogh, Picasso, Monet, Andy Warhol, Caravaggio"
contents_map["celebrity"]="Bruce Lee, Marilyn Monroe, Melania Trump, Anne Hathaway, Tom Cruise"

# 定义要使用的 GPU 索引数组
GPU_IDX=('0' '1' '2' '3' '4' '5' '6' '7')  # 指定的 GPU 数组
NUM_GPUS=${#GPU_IDX[@]}  # 计算 GPU 数量

# 初始化 GPU 分配的索引
gpu_idx=0

# 函数：提交任务到指定 GPU
run_task() {
  local erase_type=$1
  local gpu_id=$2

  echo "Running task for $erase_type on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python src/main.py \
    --erase_type "$erase_type" \
    --target_concept "$erase_type" \
    --contents "${contents_map[$erase_type]}" \
    --mode 'original' \
    --num_samples 10 --batch_size 10 \
    --save_root "data/pretrain" &  # 在后台运行任务
}

# 遍历所有任务
for erase_type in "instance" "style" "celebrity"; do
  run_task "$erase_type" ${GPU_IDX[$gpu_idx]}

  # 更新 GPU 索引，循环使用 GPU
  gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

  # 如果所有 GPU 都已分配，等待当前所有进程结束再继续
  if [ $gpu_idx -eq 0 ]; then
    wait  # 等待所有后台任务完成
  fi
done

# 等待最后一批任务完成
wait
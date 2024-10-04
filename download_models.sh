#!/bin/bash

# 定义模型及其本地路径的数组
#declare -A models=(
#  ["Qwen/Qwen-1_8B-Chat"]="/data/zwh/models/Qwen-1_8B-Chat"
##  ["Qwen/Qwen-7B-Chat"]="/data/zwh/models/Qwen-7B-Chat"
#  ["Qwen/Qwen1.5-0.5B-Chat"]="/data/zwh/models/Qwen1.5-0.5B-Chat"
#  ["Qwen/Qwen1.5-1.8B-Chat"]="/data/zwh/models/Qwen1.5-1.8B-Chat"
#  ["Qwen/Qwen1.5-4B-Chat"]="/data/zwh/models/Qwen1.5-4B-Chat"
##  ["Qwen/Qwen1.5-7B-Chat"]="/data/zwh/models/Qwen1.5-7B-Chat"
#  ["Qwen/Qwen2-0.5B-Instruct"]="/data/zwh/models/Qwen2-0.5B-Instruct"
#  ["Qwen/Qwen2-1.5B-Instruct"]="/data/zwh/models/Qwen2-1.5B-Instruct"
##  ["Qwen/Qwen2-7B-Instruct"]="/data/zwh/models/Qwen2-7B-Instruct"
#  ["Qwen/Qwen2.5-0.5B-Instruct"]="/data/zwh/models/Qwen2.5-0.5B-Instruct"
#  ["Qwen/Qwen2.5-1.5B-Instruct"]="/data/zwh/models/Qwen2.5-1.5B-Instruct"
#  ["Qwen/Qwen2.5-3B-Instruct"]="/data/zwh/models/Qwen2.5-3B-Instruct"
##  ["Qwen/Qwen2.5-7B-Instruct"]="/data/zwh/models/Qwen2.5-7B-Instruct"
#)

declare -A models=(
  ["Qwen/Qwen1.5-7B-Chat"]="/data/zwh/models/Qwen1.5-7B-Chat"
  ["Qwen/Qwen2-7B-Instruct"]="/data/zwh/models/Qwen2-7B-Instruct"
  ["Qwen/Qwen2.5-7B-Instruct"]="/data/zwh/models/Qwen2.5-7B-Instruct"
)

# 循环下载所有模型
for model in "${!models[@]}"; do
  echo "正在下载 $model ..."
  huggingface-cli download --resume-download "$model" --local-dir "${models[$model]}"
  echo "$model 下载完成！"
done

echo "所有模型下载完成！"

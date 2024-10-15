#!/bin/bash

# Qwen series
#declare -A models=(
#  ["Qwen/Qwen-1_8B-Chat"]="/data/zwh/models/Qwen-1_8B-Chat"
#  ["Qwen/Qwen-7B-Chat"]="/data/zwh/models/Qwen-7B-Chat"
#  ["Qwen/Qwen1.5-0.5B-Chat"]="/data/zwh/models/Qwen1.5-0.5B-Chat"
#  ["Qwen/Qwen1.5-1.8B-Chat"]="/data/zwh/models/Qwen1.5-1.8B-Chat"
#  ["Qwen/Qwen1.5-4B-Chat"]="/data/zwh/models/Qwen1.5-4B-Chat"
#  ["Qwen/Qwen1.5-7B-Chat"]="/data/zwh/models/Qwen1.5-7B-Chat"
#  ["Qwen/Qwen2-0.5B-Instruct"]="/data/zwh/models/Qwen2-0.5B-Instruct"
#  ["Qwen/Qwen2-1.5B-Instruct"]="/data/zwh/models/Qwen2-1.5B-Instruct"
#  ["Qwen/Qwen2-7B-Instruct"]="/data/zwh/models/Qwen2-7B-Instruct"
#  ["Qwen/Qwen2.5-0.5B-Instruct"]="/data/zwh/models/Qwen2.5-0.5B-Instruct"
#  ["Qwen/Qwen2.5-1.5B-Instruct"]="/data/zwh/models/Qwen2.5-1.5B-Instruct"
#  ["Qwen/Qwen2.5-3B-Instruct"]="/data/zwh/models/Qwen2.5-3B-Instruct"
#  ["Qwen/Qwen2.5-7B-Instruct"]="/data/zwh/models/Qwen2.5-7B-Instruct"
#)

# Pythia series
#declare -A models=(
#  ["EleutherAI/pythia-14m"]="/data2/pythia_series/pythia-14m"
#  ["EleutherAI/pythia-70m"]="/data2/pythia_series/pythia-70m"
#  ["EleutherAI/pythia-160m"]="/data2/pythia_series//pythia-160m"
#  ["EleutherAI/pythia-410m"]="/data2/pythia_series/pythia-410m"
#  ["EleutherAI/pythia-1b"]="/data2/pythia_series/pythia-1b"
#  ["EleutherAI/pythia-1.4b"]="/data2/pythia_series/pythia-1.4b"
#  ["EleutherAI/pythia-2.8b"]="/data2/pythia_series/pythia-2.8b"
#  ["EleutherAI/pythia-6.9b"]="/data2/pythia_series/pythia-6.9b"
#)

# Phi series
declare -A models=(
#  ["microsoft/phi-1"]="/data2/phi_series/phi-1"
#  ["microsoft/phi-1_5"]="/data2/phi_series/phi-1_5"
#  ["microsoft/phi-2"]="/data2/phi_series/phi-2"
  ["microsoft/Phi-3-mini-4k-instruct"]="/data2/phi_series/Phi-3-mini-4k-instruct"
  ["microsoft/Phi-3-mini-128k-instruct"]="/data2/phi_series/Phi-3-mini-128k-instruct"
  ["microsoft/Phi-3-small-8k-instruct"]="/data2/phi_series/Phi-3-small-8k-instruct"
  ["microsoft/Phi-3-small-128k-instruct"]="/data2/phi_series/Phi-3-small-128k-instruct"
#  ["microsoft/Phi-3.5-mini-instruct"]="/data2/phi_series/Phi-3.5-mini-instruct"
)

# 循环下载所有模型
for model in "${!models[@]}"; do
  echo "正在下载 $model ..."
  huggingface-cli download --resume-download "$model" --local-dir "${models[$model]}"
  echo "$model 下载完成！"
done

echo "所有模型下载完成！"

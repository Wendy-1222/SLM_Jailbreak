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
#  ["EleutherAI/pythia-31m"]="/data2/pythia_series/pythia-31m"
#  ["EleutherAI/pythia-70m"]="/data2/pythia_series/pythia-70m"
#  ["EleutherAI/pythia-160m"]="/data2/pythia_series//pythia-160m"
#  ["EleutherAI/pythia-410m"]="/data2/pythia_series/pythia-410m"
#  ["EleutherAI/pythia-1b"]="/data2/pythia_series/pythia-1b"
#  ["EleutherAI/pythia-1.4b"]="/data2/pythia_series/pythia-1.4b"
#  ["EleutherAI/pythia-2.8b"]="/data2/pythia_series/pythia-2.8b"
#  ["EleutherAI/pythia-6.9b"]="/data2/pythia_series/pythia-6.9b"
#)

# Phi series
#declare -A models=(
#  ["microsoft/phi-1"]="/data2/phi_series/phi-1"
#  ["microsoft/phi-1_5"]="/data2/phi_series/phi-1_5"
#  ["microsoft/phi-2"]="/data2/phi_series/phi-2"
#  ["microsoft/Phi-3-mini-4k-instruct"]="/data2/phi_series/Phi-3-mini-4k-instruct"
#  ["microsoft/Phi-3-mini-128k-instruct"]="/data2/phi_series/Phi-3-mini-128k-instruct"
#  ["microsoft/Phi-3-small-8k-instruct"]="/data2/phi_series/Phi-3-small-8k-instruct"
#  ["microsoft/Phi-3-small-128k-instruct"]="/data2/phi_series/Phi-3-small-128k-instruct"
#  ["microsoft/Phi-3.5-mini-instruct"]="/data2/phi_series/Phi-3.5-mini-instruct"
#)

# Cerebras-GPT series
# declare -A models=(
#   ["cerebras/Cerebras-GPT-111M"]="/data2/Cerebras-GPT_series/Cerebras-GPT-111M"
#   ["cerebras/Cerebras-GPT-256M"]="/data2/Cerebras-GPT_series/Cerebras-GPT-256M"
#   ["cerebras/Cerebras-GPT-590M"]="/data2/Cerebras-GPT_series/Cerebras-GPT-590M"
#   ["cerebras/Cerebras-GPT-1.3B"]="/data2/Cerebras-GPT_series/Cerebras-GPT-1.3B"
#   ["cerebras/Cerebras-GPT-2.7B"]="/data2/Cerebras-GPT_series/Cerebras-GPT-2.7B"
#   ["cerebras/Cerebras-GPT-6.7B"]="/data2/Cerebras-GPT_series/Cerebras-GPT-6.7B"
# )

# # StableLM series
# declare -A models=(
#   ["stabilityai/stablelm-2-zephyr-1_6b"]="/data2/stablelm_series/stablelm-2-zephyr-1_6b"
#   ["stabilityai/stablelm-zephyr-3b"]="/data2/stablelm_series/stablelm-zephyr-3b"
#   ["stabilityai/stablelm-2-1_6b-chat"]="/data2/stablelm_series/stablelm-2-1_6b-chat"
# )

# TinyLlama series
#declare -A models=(
#  ["TinyLlama/TinyLlama_v1.1"]="/data2/TinyLlama/TinyLlama_v1.1"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v0.1"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v0.1"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v0.2"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v0.2"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v0.3"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v0.3"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v0.4"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v0.4"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v0.5"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v0.5"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v0.6"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v0.6"
#  ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]="/data2/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#)

# # MobileLLaMA series
# declare -A models=(
#   ["mtgv/MobileLLaMA-1.4B-Chat"]="/data2/MobileLLaMA/MobileLLaMA-1.4B-Chat"
#   ["mtgv/MobileLLaMA-2.7B-Chat"]="/data2/MobileLLaMA/MobileLLaMA-2.7B-Chat"
# )

# # MobiLlama series
# declare -A models=(
#   ["MBZUAI/MobiLlama-05B-Chat"]="/data2/MobiLlama/MobiLlama-0.5B-Chat"
#   ["MBZUAI/MobiLlama-1B-Chat"]="/data2/MobiLlama/MobiLlama-1B-Chat"
# )

# # LaMini-LM series
# declare -A models=(
#   ["MBZUAI/LaMini-T5-61M"]="/data2/LaMini-LM/LaMini-T5-61M"
#   ["MBZUAI/LaMini-T5-223M"]="/data2/LaMini-LM/LaMini-T5-223M"
#   ["MBZUAI/LaMini-T5-738M"]="/data2/LaMini-LM/LaMini-T5-738M"
#   ["MBZUAI/LaMini-Flan-T5-77M"]="/data2/LaMini-LM/LaMini-Flan-T5-77M"
#   ["MBZUAI/LaMini-Flan-T5-248M"]="/data2/LaMini-LM/LaMini-Flan-T5-248M"
#   ["MBZUAI/LaMini-Flan-T5-783M"]="/data2/LaMini-LM/LaMini-Flan-T5-783M"
#   ["MBZUAI/LaMini-Cerebras-111M"]="/data2/LaMini-LM/LaMini-Cerebras-111M"
#   ["MBZUAI/LaMini-Cerebras-256M"]="/data2/LaMini-LM/LaMini-Cerebras-256M"
#   ["MBZUAI/LaMini-Cerebras-590M"]="/data2/LaMini-LM/LaMini-Cerebras-590M"
#   ["MBZUAI/LaMini-Cerebras-1.3B"]="/data2/LaMini-LM/LaMini-Cerebras-1.3B"
#   ["MBZUAI/LaMini-GPT-124M"]="/data2/LaMini-LM/LaMini-GPT-124M"
#   ["MBZUAI/LaMini-GPT-774M"]="/data2/LaMini-LM/LaMini-GPT-774M"
#   ["MBZUAI/LaMini-GPT-1.5B"]="/data2/LaMini-LM/LaMini-GPT-1.5B"
#   ["MBZUAI/LaMini-Neo-125M"]="/data2/LaMini-LM/LaMini-Neo-125M"
#   ["MBZUAI/LaMini-Neo-1.3B"]="/data2/LaMini-LM/LaMini-Neo-1.3B"
# )


# # Gemma series
# declare -A models=(
#  ["google/gemma-2b-it"]="/data2/gemma_series/gemma-2b-it"
#  ["google/gemma-7b-it"]="/data2/gemma_series/gemma-7b-it"
#  ["google/gemma-1.1-2b-it"]="/data2/gemma_series/gemma-1.1-2b-it"
#  ["google/gemma-1.1-7b-it"]="/data2/gemma_series/gemma-1.1-7b-it"
#  ["google/gemma-2-2b-it"]="/data2/gemma_series/gemma-2-2b-it"
#  ["google/recurrentgemma-2b-it"]="/data2/gemma_series/recurrentgemma-2b-it"
# )

# # MiniCPM series
# declare -A models=(
#   ["openbmb/MiniCPM-1B-sft-bf16"]="/data2/MiniCPM_series/MiniCPM-1B-sft-bf16"
#   ["openbmb/MiniCPM-S-1B-sft"]="/data2/MiniCPM_series/MiniCPM-S-1B-sft"
#   ["openbmb/MiniCPM-2B-128k"]="/data2/MiniCPM_series/MiniCPM-2B-128k"
#   ["openbmb/MiniCPM-2B-dpo-int4"]="/data2/MiniCPM_series/MiniCPM-2B-dpo-int4"
#   ["openbmb/MiniCPM-2B-dpo-fp16"]="/data2/MiniCPM_series/MiniCPM-2B-dpo-fp16"
#   ["openbmb/MiniCPM-2B-dpo-bf16"]="/data2/MiniCPM_series/MiniCPM-2B-dpo-bf16"
#   ["openbmb/MiniCPM-2B-dpo-fp32"]="/data2/MiniCPM_series/MiniCPM-2B-dpo-fp32"
#   ["openbmb/MiniCPM-2B-sft-int4"]="/data2/MiniCPM_series/MiniCPM-2B-sft-int4"
#   ["openbmb/MiniCPM-2B-sft-bf16"]="/data2/MiniCPM_series/MiniCPM-2B-sft-bf16"
#   ["openbmb/MiniCPM-2B-sft-fp32"]="/data2/MiniCPM_series/MiniCPM-2B-sft-fp32"
#   ["openbmb/MiniCPM3-4B"]="/data2/MiniCPM_series/MiniCPM3-4B"
# )

# # CPM-Bee series
# declare -A models=(
#   ["openbmb/cpm-bee-1b"]="/data2/CPM-Bee_series/cpm-bee-1b"
#   ["openbmb/cpm-bee-2b"]="/data2/CPM-Bee_series/cpm-bee-2b"
#   ["openbmb/cpm-bee-5b"]="/data2/CPM-Bee_series/cpm-bee-5b"
# )

# # OpenELM series
# declare -A models=(
#   ["apple/OpenELM-270M-Instruct"]="/data2/OpenELM/OpenELM-270M-Instruct"
#   ["apple/OpenELM-450M-Instruct"]="/data2/OpenELM/OpenELM-450M-Instruct"
#   ["apple/OpenELM-1_1B-Instruct"]="/data2/OpenELM/OpenELM-1_1B-Instruct"
#   ["apple/OpenELM-3B-Instruct"]="/data2/OpenELM/OpenELM-3B-Instruct"
# )

# # danube series
# declare -A models=(
#   ["h2oai/h2o-danube-1.8b-sft"]="/data2/danube_series/h2o-danube-1.8b-sft"
#   ["h2oai/h2o-danube-1.8b-chat"]="/data2/danube_series/h2o-danube-1.8b-chat"
#   ["h2oai/h2o-danube2-1.8b-sft"]="/data2/danube_series/h2o-danube2-1.8b-sft"
#   ["h2oai/h2o-danube2-1.8b-chat"]="/data2/danube_series/h2o-danube2-1.8b-chat"
#   ["h2oai/h2o-danube3-500m-chat"]="/data2/danube_series/h2o-danube3-500m-chat"
#   ["h2oai/h2o-danube3-4b-chat"]="/data2/danube_series/h2o-danube3-4b-chat"
# )

# # SmolLM series
# declare -A models=(
#   ["HuggingFaceTB/SmolLM-135M-Instruct"]="/data2/SmolLM_series/SmolLM-135M-Instruct"
#   ["HuggingFaceTB/SmolLM-360M-Instruct"]="/data2/SmolLM_series/SmolLM-360M-Instruct"
#   ["HuggingFaceTB/SmolLM-1.7B-Instruct"]="/data2/SmolLM_series/SmolLM-1.7B-Instruct"
# )

# SmolLM series
#declare -A models=(
#  ["HuggingFaceTB/SmolLM2-135M-Instruct"]="/data2/SLMs/SmolLM_series/SmolLM2-135M-Instruct"
#  ["HuggingFaceTB/SmolLM2-360M-Instruct"]="/data2/SLMs/SmolLM_series/SmolLM2-360M-Instruct"
#  ["HuggingFaceTB/SmolLM2-1.7B-Instruct"]="/data2/SLMs/SmolLM_series/SmolLM2-1.7B-Instruct"
#)

# # DCLM series
# declare -A models=(
#   ["TRI-ML/DCLM-1B-IT"]="/data2/DCLM_series/DCLM-1B-IT"
# )

# # Dolly series
# declare -A models=(
#   ["databricks/dolly-v1-6b"]="/data2/dolly_series/dolly-v1-6b"
#   ["databricks/dolly-v2-3b"]="/data2/dolly_series/dolly-v2-3b"
#   ["databricks/dolly-v2-7b"]="/data2/dolly_series/dolly-v2-7b"
# )

# # OLMo series
# declare -A models=(
#   ["allenai/OLMo-1B-hf"]="/data2/OLMo_series/OLMo-1B-hf"
#   ["allenai/OLMo-7B-hf"]="/data2/OLMo_series/OLMo-7B-hf"
#   ["allenai/OLMo-7B-SFT-hf"]="/data2/OLMo_series/OLMo-7B-SFT-hf"
#   ["allenai/OLMo-7B-Instruct-hf"]="/data2/OLMo_series/OLMo-7B-Instruct-hf"
# )

# # llama-3.2
# declare -A models=(
#   ["meta-llama/Llama-3.1-8B-Instruct"]="/data2/llama3.1/Llama-3.1-8B-Instruct"
# )

# declare -A models=(
#   ["tensoropera/Fox-1-1.6B-Instruct-v0.1"]="/data2/SLMs/fox_series/Fox-1-1.6B-Instruct-v0.1"
#)

#declare -A models=(
#    ["meta-llama/Llama-Guard-3-8B"]="/data2/llama3.1/Llama-Guard-3-8B"
#)

#declare -A models=(
#    ["meta-llama/Llama-3.2-1B-Instruct"]="/data2/llama3.2/Llama-3.2-1B-Instruct"
#    ["meta-llama/Llama-3.2-3B-Instruct"]="/data2/llama3.2/Llama-3.2-3B-Instruct"
#)

# declare -A models=(
#     ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]="/data2/SLMs/Deepseek-R1/DeepSeek-R1-Distill-Qwen-1.5B"
#     ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]="/data2/SLMs/Deepseek-R1/DeepSeek-R1-Distill-Qwen-7B"
#     ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="/data2/SLMs/Deepseek-R1/DeepSeek-R1-Distill-Llama-8B"
# )

# declare -A models=(
#     ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4"]="/data2/SLMs/qwen_series/Qwen1.5-0.5B-Chat-GPTQ-Int4"
#     ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"]="/data2/SLMs/qwen_series/Qwen1.5-7B-Chat-GPTQ-Int4"
# )

# declare -A models=(
#     ["meta-llama/Llama-Guard-3-1B"]="/data2/zwh/models/Llama-Guard-3-1B"
#     ["meta-llama/Prompt-Guard-86M"]="/data2/zwh/models/Prompt-Guard-86M"
# )

# declare -A models=(
#     ["openbmb/MiniCPM-V-2_6"]="/data2/zwh/models/MiniCPM-V-2_6"
#     ["openbmb/MiniCPM-V-2_6-int4"]="/data2/zwh/models/MiniCPM-V-2_6-int4"
# )

declare -A models=(
    ["Qwen/Qwen2-VL-2B-Instruct"]="/data2/zwh/models/Qwen2-VL-2B-Instruct"
    ["Qwen/Qwen2-VL-2B-Instruct-AWQ"]="/data2/zwh/models/Qwen2-VL-2B-Instruct-AWQ"
    ["Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"]="/data2/zwh/models/Qwen2-VL-2B-Instruct-GPTQ-Int4"
    ["Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8"]="/data2/zwh/models/Qwen2-VL-2B-Instruct-GPTQ-Int8"
    ["Qwen/Qwen2-VL-7B-Instruct"]="/data2/zwh/models/Qwen2-VL-7B-Instruct"
    ["Qwen/Qwen2-VL-7B-Instruct-AWQ"]="/data2/zwh/models/Qwen2-VL-7B-Instruct-AWQ"
    ["Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"]="/data2/zwh/models/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    ["Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8"]="/data2/zwh/models/Qwen2-VL-7B-Instruct-GPTQ-Int8"
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="/data2/zwh/models/Qwen2.5-VL-3B-Instruct"
    ["Qwen/Qwen2.5-VL-3B-Instruct-AWQ"]="/data2/zwh/models/Qwen2.5-VL-3B-Instruct-AWQ"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="/data2/zwh/models/Qwen2.5-VL-7B-Instruct"
    ["Qwen/Qwen2.5-VL-7B-Instruct-AWQ"]="/data2/zwh/models/Qwen2.5-VL-7B-Instruct-AWQ"
)


# 循环下载所有模型
for model in "${!models[@]}"; do
  echo "正在下载 $model ..."
  huggingface-cli download --resume-download "$model" --local-dir "${models[$model]}"
  echo "$model 下载完成！"
done

echo "所有模型下载完成！"

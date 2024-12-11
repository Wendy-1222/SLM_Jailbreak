import os

# qwen_series = "qwen_1_8b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct"
# tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"
# mobi_llama_series = "mobillama-0.5B-chat,mobillama-1B-chat"
#
# slms = qwen_series + ',' + tiny_llama_series + ',' + mobi_llama_series

# minicpm_series = "minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-dpo-bf16,minicpm-2B-128k,minicpm3-4B,fox-1-1.6B-Instruct-v0.1"
# phi_series = "phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_5_mini_instruct"
# stablelm_series = "stablelm-2-zephyr-1_6b,stablelm-2-1_6b-chat,stablelm-zephyr-3b"
# h2o_danube_series = "h2o-danube2-1.8b-chat,h2o-danube-1.8b-sft,h2o-danube-1.8b-chat,h2o-danube2-1.8b-sft,h2o-danube3-500m-chat"
# smollm_series = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"
#
# slms = minicpm_series + ',' + phi_series + ',' + stablelm_series + ',' + h2o_danube_series + ',' + smollm_series

slms="llama2_7b,vicuna_7b_v1_5,qwen_7b_chat,qwen1_5_7b_chat,qwen2_7b_instruct,qwen2_5_7b_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,mobilellama-1.4B-chat,mobilellama-2.7B-chat,dolly-v1-6b,dolly-v2-3b,dolly-v2-7b,OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"

# gemma_series = "gemma-2b-it,gemma-7b-it,gemma-1.1-2b-it,gemma-1.1-7b-it,gemma-2-2b-it,recurrentgemma-2b-it"
# slms = "phi_1_5,phi_2" + ',' + gemma_series

# methods="DirectRequest,HumanJailbreaks,PEZ,UAT,GBDA"  # or "all" to use all methods
# methods = "PAP-top5"
# methods = "DirectRequest,HumanJailbreaks"
methods = "PEZ"
models = "mobilellama-2.7B-chat"
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="2_and_3"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

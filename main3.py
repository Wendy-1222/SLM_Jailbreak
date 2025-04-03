import os

# base_save_dir="./results"
# base_log_dir="./slurm_logs"
other_series = "llama2_7b,vicuna_7b_v1_5"
qwen_series = "qwen_1_8b_chat,qwen_7b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen1_5_7b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_7b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct,qwen2_5_7b_instruct"
pythia_series = "pythia_14m,pythia_31m,pythia_70m,pythia_160m,pythia_410m,pythia_1b,pythia_1_4b,pythia_2_8b,pythia_6_9b"
phi_series = "phi_1_5,phi_2,phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,phi_3_5_mini_instruct"
stablelm_series = "stablelm-2-zephyr-1_6b,stablelm-2-1_6b-chat,stablelm-zephyr-3b"
tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"
mobile_llama_series = "mobilellama-1.4B-chat,mobilellama-2.7B-chat"
mobi_llama_series = "mobillama-0.5B-chat,mobillama-1B-chat"
gemma_series = "gemma-2b-it,gemma-7b-it,gemma-1.1-2b-it,gemma-1.1-7b-it,gemma-2-2b-it,recurrentgemma-2b-it"
minicpm_series = "minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-sft-fp32,minicpm-2B-sft-int4,minicpm-2B-dpo-bf16,minicpm-2B-dpo-fp16,minicpm-2B-dpo-fp32,minicpm-2B-dpo-int4,minicpm-2B-128k,minicpm3-4B"
h2o_danube_series = "h2o-danube-1.8b-sft,h2o-danube-1.8b-chat,h2o-danube2-1.8b-sft,h2o-danube2-1.8b-chat,h2o-danube3-500m-chat,h2o-danube3-4b-chat"
fox_series = "fox-1-1.6B-Instruct-v0.1"
smollm_series = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"
dclm_series = "DCLM-1B-IT"
dolly_series = "dolly-v1-6b,dolly-v2-3b,dolly-v2-7b"
olmo_series = "OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"
llama_3_2_series = "llama3_2_1b_instruct,llama3_2_3b_instruct"
deepseek_r1_series = "DeepSeek-R1-Distill-Qwen-1.5B,DeepSeek-R1-Distill-Qwen-7B"
# deepseek_r1_series = "DeepSeek-R1-Distill-Qwen-1.5B"

all_slms = other_series + ',' + qwen_series + ',' + phi_series + ',' + stablelm_series + ',' + tiny_llama_series + ',' + mobile_llama_series + ',' + mobi_llama_series + ',' + gemma_series + ',' + minicpm_series + ',' + h2o_danube_series + ',' + fox_series + ',' + smollm_series + ',' + dolly_series + ',' + olmo_series + ',' + dclm_series + ',' + llama_3_2_series + ',' +  deepseek_r1_series
models_7B = "DeepSeek-R1-Distill-Qwen-7B,qwen_7b_chat,qwen1_5_7b_chat,qwen2_7b_instruct,qwen2_5_7b_instruct,gemma-7b-it,gemma-1.1-7b-it,dolly-v1-6b,dolly-v2-7b,OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"
models_4B = "llama3_2_3b_instruct,phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_5_mini_instruct,minicpm3-4B,mobilellama-2.7B-chat,qwen1_5_4b_chat,qwen2_5_3b_instruct,stablelm-zephyr-3b,h2o-danube3-4b-chat"
all_slms_list = all_slms.split(',')
models_list_7B = models_7B.split(',')
models_list_4B = models_4B.split(',')
all_slms_list_no_large = [item for item in all_slms_list if (item not in models_list_7B and item not in models_list_4B)]
all_slms_no_large = ','.join(all_slms_list_no_large)

# mid_index = len(all_slms_list_no_large) // 2
# first_half = all_slms_list_no_large[:mid_index]
# second_half = all_slms_list_no_large[mid_index:]
# first_half = ','.join(first_half)
# second_half = ','.join(second_half)

# # 忽略了目前template有点问题的模型
# all_slms = qwen_series + ',' + stablelm_series + ',' + mobile_llama_series + ',' + mobi_llama_series + ',' + gemma_series + ',' + h2o_danube_series + ',' + fox_series + ',' + dolly_series + ',' + olmo_series + ',' + dclm_series

methods="DirectRequest,PAP-top5,HumanJailbreaks,AutoDAN,GCG,AutoPrompt,PEZ,UAT,GBDA"  # or "all" to use all methods
# methods = "AutoDAN"
models = mobi_llama_series
# models = "qwen1_5_0_5b_chat"
defender = "llama_guard_3"
# behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="4_and_5"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

# 注意defender和incremental_update，还有save_dir
os.system(f"python ./scripts/run_pipeline.py --defender {defender} --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")
# os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50 --defender {defender} --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")


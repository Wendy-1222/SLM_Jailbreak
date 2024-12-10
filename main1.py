import os

# base_save_dir="./results"
# base_log_dir="./slurm_logs"
other_series = "llama2_7b,vicuna_7b_v1_5"
qwen_series = "qwen_1_8b_chat,qwen_7b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen1_5_7b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_7b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct,qwen2_5_7b_instruct"
pythia_series = "pythia_14m,pythia_31m,pythia_70m,pythia_160m,pythia_410m,pythia_1b,pythia_1_4b,pythia_2_8b,pythia_6_9b"
phi_series = "phi_1_5,phi_2,phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,phi_3_5_mini_instruct"
stablelm_series = "stablelm-2-zephyr-1_6b,stablelm-2-1_6b-chat,stablelm-zephyr-3b"
tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"
mobile_llama_series="mobilellama-1.4B-chat,mobilellama-2.7B-chat"
mobi_llama_series="mobillama-0.5B-chat,mobillama-1B-chat"
minicpm_series="minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-sft-fp32,minicpm-2B-sft-int4,minicpm-2B-dpo-bf16,minicpm-2B-dpo-fp16,minicpm-2B-dpo-fp32,minicpm-2B-dpo-int4,minicpm-2B-128k,minicpm3-4B"
h2o_danube_series = "h2o-danube2-1.8b-chat,h2o-danube-1.8b-sft,h2o-danube-1.8b-chat,h2o-danube2-1.8b-sft,h2o-danube3-500m-chat"
fox_series = "fox-1-1.6B-Instruct-v0.1"
smollm_series = "smollm2-1.7B-instruct,smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct"
dolly_series = "dolly-v1-6b,dolly-v2-3b,dolly-v2-7b"
olmo_series = "OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"
dclm_series = "DCLM-1B-IT"

# slms = "qwen_1_8b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct"
# slms = "minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-dpo-bf16,minicpm-2B-128k,minicpm3-4B,fox-1-1.6B-Instruct-v0.1"
slms="llama2_7b,vicuna_7b_v1_5,qwen_7b_chat,qwen1_5_7b_chat,qwen2_7b_instruct,qwen2_5_7b_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,mobilellama-1.4B-chat,mobilellama-2.7B-chat,dolly-v1-6b,dolly-v2-3b,dolly-v2-7b,OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"

methods="DirectRequest,HumanJailbreaks,PEZ,UAT,GBDA"
models = "h2o-danube3-4b-chat"
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="1.5"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")


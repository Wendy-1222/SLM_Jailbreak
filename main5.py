import os

models_7b_1 = "llama2_7b,vicuna_7b_v1_5,qwen_7b_chat,qwen1_5_7b_chat,qwen2_7b_instruct,qwen2_5_7b_instruct"
qwen_1 = "qwen1_5_0_5b_chat,qwen2_0_5b_instruct,qwen2_5_0_5b_instruct,qwen1_5_4b_chat,qwen_1_8b_chat,qwen1_5_1_8b_chat,qwen2_1_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct"
mobile_llama_series = "mobilellama-1.4B-chat,mobilellama-2.7B-chat"
mobi_llama_series = "mobillama-0.5B-chat,mobillama-1B-chat"
minicpm_series = "minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-dpo-bf16"

methods="AutoPrompt"
models = minicpm_series
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

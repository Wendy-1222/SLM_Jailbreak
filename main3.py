import os

slms="llama2_7b,vicuna_7b_v1_5,qwen_7b_chat,qwen1_5_7b_chat,qwen2_7b_instruct,qwen2_5_7b_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,dolly-v2-7b,OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"

methods="DirectRequest,HumanJailbreaks,PEZ,UAT,GBDA"
models = "mobilellama-2.7B-chat"
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="1.5"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

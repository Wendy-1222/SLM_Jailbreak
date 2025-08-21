import os

llama_series = "llama2_7b,llama3_1_8b_instruct,llama3_2_1b_instruct,llama3_2_3b_instruct"
qwen_series = "qwen_1_8b_chat,qwen_7b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen1_5_7b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_7b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct,qwen2_5_7b_instruct"
smollm_series = "smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"
slms_with_system_prompt = llama_series + ',' + qwen_series + ',' + smollm_series

methods="PEZ,UAT,GBDA"  # or "all" to use all methods
# methods = "AutoPrompt,PEZ,UAT,GBDA"
models = slms_with_system_prompt
defender = "self-reminder-with-system-prompt"
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


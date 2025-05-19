import os

methods = "DirectRequest,HumanJailbreaks,PAP-top5"
models = "qwen1_5_0_5b_chat_gptq_int4"
# step="4"
step="2_and_3"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# 注意defender和incremental_update，还有save_dir
os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50 --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
# 注意defender和incremental_update，还有save_dir
os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

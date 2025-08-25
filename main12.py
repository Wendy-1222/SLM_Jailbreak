import os

methods = "GBDA"
models = "llama3_2_1b_instruct"
step="1"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# 注意defender和incremental_update，还有save_dir
os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50_for_cost_analysis --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

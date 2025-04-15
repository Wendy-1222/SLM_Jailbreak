import os

methods = "HumanJailbreaks"
# models = "minicpm-S-1B-sft"
models = "dolly-v1-6b"
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step = "all"  # or "1", "1.5", "2", "3", "2_and_3"
mode = "local"
# mode="slurm"
# partition="your_partition"
cls_path = "/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50 --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")
# os.system(f"python ./scripts/run_pipeline.py --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")


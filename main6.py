import os

methods="GCG"  # or "all" to use all methods
models="tinyllama-1.1B-chat-v0.5"  # or "all" to use all models
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path="./data/behavior_datasets/extra_behavior_datasets/temp.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

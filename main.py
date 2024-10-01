import os

# base_save_dir="./results"
# base_log_dir="./slurm_logs"
methods="PAIR"  # or "all" to use all methods
models="llama2_7b"  # or "all" to use all models
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local_parallel"
# mode="slurm"
# partition="your_partition"
# cls_path="cais/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode}")


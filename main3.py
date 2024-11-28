import os

smollm_series = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"
dclm_series = "DCLM-1B-IT"
dolly_series = "dolly-v1-6b,dolly-v2-3b,dolly-v2-7b"

methods="GCG"  # or "all" to use all methods
models = "OLMo-7B-Instruct-hf"
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path="./data/behavior_datasets/extra_behavior_datasets/temp.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

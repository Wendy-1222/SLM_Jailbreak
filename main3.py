import os

mobile_llama_series="mobilellama-1.4B-chat,mobilellama-2.7B-chat"
mobi_llama_series="mobillama-0.5B-chat,mobillama-1B-chat"

methods="GCG"
models="mobilellama-1.4B-chat"
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path="./data/behavior_datasets/extra_behavior_datasets/temp.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

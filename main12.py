import os

h2o_danube_series = "h2o-danube-1.8b-sft,h2o-danube-1.8b-chat,h2o-danube2-1.8b-sft,h2o-danube2-1.8b-chat,h2o-danube3-500m-chat,h2o-danube3-4b-chat"
fox_series = "fox-1-1.6B-Instruct-v0.1"
smollm_series = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"

methods="AutoPrompt"  # or "all" to use all methods
models="smollm2-360M-instruct"
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path="./data/behavior_datasets/extra_behavior_datasets/temp.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

import os

methods = "UAT"
models = "OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"
# models = "gemma-2-2b-it" + ',' + tiny_series
defender_list = ["ppl", "self-reminder", "retokenization", "llama_guard_3", "llama_guard_3_1B"]
# step="4"
step="4_and_5"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
for defender in defender_list:
    # 注意defender和incremental_update，还有save_dir
    os.system(f"python ./scripts/run_pipeline.py --defender {defender} --base_save_dir ./results_full_50 --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
for defender in defender_list:
    # 注意defender和incremental_update，还有save_dir
    os.system(f"python ./scripts/run_pipeline.py --defender {defender} --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

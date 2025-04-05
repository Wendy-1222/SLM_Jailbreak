import os

tiny_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6"
need_slms = "dolly-v1-6b" + ',' + "OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf" + ',' + "gemma-2-2b-it" + ',' + tiny_series

methods = "UAT"
models = need_slms
# models = "gemma-2-2b-it" + ',' + tiny_series
defender_list = ["ppl", "self-reminder", "retokenization", "llama_guard_3", "llama_guard_3_1B"]
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
# step="4"
step="4_and_5"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

for defender in defender_list:
    # 注意defender和incremental_update，还有save_dir
    # os.system(f"python ./scripts/run_pipeline.py --defender {defender} --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")
    os.system(f"python ./scripts/run_pipeline.py --defender {defender} --base_save_dir ./results_full_50 --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

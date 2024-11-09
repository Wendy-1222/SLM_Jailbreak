import os

stablelm_series = "stablelm-2-zephyr-1_6b,stablelm-2-1_6b-chat,stablelm-zephyr-3b"
tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"

methods="AutoPrompt"  # or "all" to use all methods
models="tinyllama-1.1B-chat-v0.6"  # or "all" to use all models
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path="./data/behavior_datasets/extra_behavior_datasets/temp.csv"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

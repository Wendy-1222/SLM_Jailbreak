import os

tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"
mobi_llama_series = "mobillama-0.5B-chat,mobillama-1B-chat"

slms = tiny_llama_series + ',' + mobi_llama_series

methods="DirectRequest,HumanJailbreaks,PEZ,UAT,GBDA"  # or "all" to use all methods
models = slms
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="1"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

os.system(f"python ./scripts/run_pipeline.py --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

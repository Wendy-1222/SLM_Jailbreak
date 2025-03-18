import os

models_7B = "qwen_7b_chat,qwen1_5_7b_chat,qwen2_7b_instruct,qwen2_5_7b_instruct,gemma-7b-it,gemma-1.1-7b-it,dolly-v1-6b,dolly-v2-7b,OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"
models_4B = "phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_5_mini_instruct,minicpm3-4B,mobilellama-2.7B-chat,qwen1_5_4b_chat,qwen2_5_3b_instruct,stablelm-zephyr-3b,h2o-danube3-4b-chat"

methods = "AutoDAN"
# models = "phi_3_mini_128k_instruct,phi_3_5_mini_instruct"
# models = "minicpm3-4B,mobilellama-2.7B-chat,qwen1_5_4b_chat,qwen2_5_3b_instruct,stablelm-zephyr-3b,h2o-danube3-4b-chat,phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_5_mini_instruct"
models = "qwen2_5_3b_instruct,stablelm-zephyr-3b,h2o-danube3-4b-chat,phi_3_mini_4k_instruct"
# behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step = "1"  # or "1", "1.5", "2", "3", "2_and_3"
mode = "local"
# mode="slurm"
# partition="your_partition"
cls_path = "/data/zwh/models/HarmBench-Llama-2-13b-cls"

# os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50 --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")
os.system(f"python ./scripts/run_pipeline.py --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")


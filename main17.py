import os

methods = "AutoDAN"
models = "dolly-v1-6b"
behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# behaviors_path = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
step="4_and_5"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

for defender in ['ppl', 'self-reminder', 'retokenization', 'llama_guard_3']:
    # 注意defender和incremental_update，还有save_dir
    # os.system(f"python ./scripts/run_pipeline.py --defender {defender} --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")
    os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50 --defender {defender} --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

os.system('python main1.py')
os.system('python main.py')
os.system('python main2.py')
os.system('python main3.py')

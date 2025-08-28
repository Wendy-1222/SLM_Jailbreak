# for attack

import os

qwen2_5_quantized_models_int4 = "qwen2_5_0_5b_instruct_gptq_int4,qwen2_5_1_5b_instruct_gptq_int4,qwen2_5_3b_instruct_gptq_int4,qwen2_5_7b_instruct_gptq_int4"
qwen2_5_quantized_models_int8 = "qwen2_5_0_5b_instruct_gptq_int8,qwen2_5_1_5b_instruct_gptq_int8,qwen2_5_3b_instruct_gptq_int8,qwen2_5_7b_instruct_gptq_int8"
qwen2_5_awq_models = "qwen2_5_0_5b_instruct_awq,qwen2_5_1_5b_instruct_awq,qwen2_5_3b_instruct_awq,qwen2_5_7b_instruct_awq"

methods = "GCG"
models = "phi_3_small_8k_instruct"
step="all"  # or "1", "1.5", "2", "3", "2_and_3"
mode="local"
# mode="slurm"
# partition="your_partition"
cls_path="/data/zwh/models/HarmBench-Llama-2-13b-cls"

behaviors_path="./data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv"
# Note defender, incremental_update, andsave_dir
os.system(f"python ./scripts/run_pipeline.py --base_save_dir ./results_full_50 --incremental_update --methods {methods} --models {models} --behaviors_path {behaviors_path} --step {step} --mode {mode} --cls_path {cls_path}")

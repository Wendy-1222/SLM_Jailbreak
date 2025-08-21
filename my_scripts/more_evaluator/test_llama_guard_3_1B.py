"""
使用Llama-Guard-3-1B进行评分
https://huggingface.co/meta-llama/Llama-Guard-3-1B
"""

import json
import os
import sys

sys.path.append('/data2/zwh/HarmBench')

from defense_baselines.LlamaGuard3.detect import Llama3_guard_Predictor
from model_and_method_list import model_list, method_list

# 处理一个project的所有run，返回一个新的df
def process_project(completions_path):
    # 读取json文件
    with open(completions_path, 'r') as f:
        completions_data = json.load(f)

    results = {}
    for run_id, run_data in completions_data.items():   
        response_list = [case["generation"] for case in run_data]
        guard_raw_outputs_list, label_list, category_list = llama_guard_3.predict(response_list)  
        
        # 为每个generation添加raw_output、label和category
        processed_data = []
        for case, guard_raw_output, label, category in zip(run_data, guard_raw_outputs_list, label_list, category_list):
            processed_case = case.copy()
            processed_case["raw_output"] = guard_raw_output
            processed_case["label"] = label
            processed_case["category"] = category
            processed_data.append(processed_case)

        results[run_id] = processed_data

    return results

base_path = '/data2/zwh/HarmBench/results/'
results_summary = {}

llama_guard_3 = Llama3_guard_Predictor('/data2/zwh/models/Llama-Guard-3-1B')
print("Done loading model")

method_list = ["AutoDAN"]
model_list = ["qwen2_5_1_5b_instruct_gptq_int4", "qwen2_5_3b_instruct_gptq_int4", "qwen2_5_7b_instruct_gptq_int4", "qwen2_5_0_5b_instruct_gptq_int8", "qwen2_5_1_5b_instruct_gptq_int8", "qwen2_5_3b_instruct_gptq_int8", "qwen2_5_7b_instruct_gptq_int8"]

for method in method_list:
    print("Processing method {}".format(method))
    results_summary[method] = {}
    for model in model_list:
        results_summary[method][model] = {}
        
        # 获得completions的路径
        if method == 'DirectRequest':
            completions_path = os.path.join(base_path, method, 'default', 'completions', f'{model}.json')
        elif method == 'HumanJailbreaks':
            completions_path = os.path.join(base_path, method, 'random_subset_5', 'completions', f'{model}.json')
        elif method == 'PAP':
            completions_path = os.path.join(base_path, method, 'top_5', 'completions', f'{model}.json')
        else:
            completions_path =  os.path.join(base_path, method, model, 'completions', f'{model}.json')
        completions_path = completions_path.replace("\\", "/")

        if not os.path.exists(completions_path):
            print(f"File {completions_path} not found, skipping...")
            continue

        # 修改路径为 results 并更改文件名
        result_path = completions_path.replace("/completions/", "/results_Llama-Guard-3-1B/")
        # 如果已存在，则跳过
        if os.path.exists(result_path):
            print(f"File {result_path} already exists, skipping...")
            continue
        else:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

        project_results = process_project(completions_path)
        with open(result_path, 'w') as f:
            json.dump(project_results, f, indent=4)

        print(f"Processed results saved to {result_path}")

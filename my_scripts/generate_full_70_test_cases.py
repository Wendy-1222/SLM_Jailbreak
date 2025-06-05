"""
从50个问题的结果里面选30个，加上40个问题的结果，得到70个问题的结果
"""

import os
import json
import argparse

from model_and_method_list import method_list, model_list

base_path = '/data2/zwh/HarmBench/results_full_50/'
base_path_added = '/data2/zwh/HarmBench/results/'
target_path = '/data2/zwh/HarmBench/results_full_70/'

in_70_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 23, 25, 26, 28, 31, 32, 38, 39, 41, 42,
              47]
in_70_key = [f'advbench_subset_{i}' for i in in_70_list]

def process_test_cases(method, model, parent_path):
    # 处理test_cases_individual_behaviors目录，如果存在的话
    test_cases_individual_behaviors_path = os.path.join(parent_path, 'test_cases', 'test_cases_individual_behaviors')
    test_cases_individual_behaviors_path_added = test_cases_individual_behaviors_path.replace(base_path,
                                                                                              base_path_added)

    if os.path.exists(test_cases_individual_behaviors_path) and os.path.exists(test_cases_individual_behaviors_path_added):
        # 里面有很多子目录，把子目录名称在in_70_key里的都复制到target_path的对应目录下
        test_cases_individual_behaviors_path_full_70 = test_cases_individual_behaviors_path.replace(base_path, target_path)
        os.makedirs(test_cases_individual_behaviors_path_full_70, exist_ok=True)
        for subdir in os.listdir(test_cases_individual_behaviors_path):
            if subdir in in_70_key:
                subdir_path = os.path.join(test_cases_individual_behaviors_path, subdir)
                target_subdir_path = os.path.join(test_cases_individual_behaviors_path_full_70, subdir)
                os.system(f"cp -r {subdir_path} {target_subdir_path}")

        # added 里面所有的子目录都复制到target_path的对应目录下
        for subdir in os.listdir(test_cases_individual_behaviors_path_added):
            subdir_path = os.path.join(test_cases_individual_behaviors_path_added, subdir)
            target_subdir_path = test_cases_individual_behaviors_path_full_70 # 不需要subdir
            # print("subdir:", subdir)
            # print("subdir_path:", subdir_path)
            # print("target_subdir_path:", target_subdir_path)
            os.system(f"cp -r {subdir_path} {target_subdir_path}")
    else:
        print(f"{method} {model} test cases individual behaviors directory not found, skipping...")

    # 处理test_cases.json文件
    test_cases_path = os.path.join(parent_path, 'test_cases', 'test_cases.json')
    test_cases_path_added = test_cases_path.replace(base_path, base_path_added)

    if os.path.exists(test_cases_path) and os.path.exists(test_cases_path_added):
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        with open(test_cases_path_added, 'r', encoding='utf-8') as f:
            test_cases_added = json.load(f)

        full_test_cases = {k: v for k, v in test_cases.items() if k in in_70_key}
        full_test_cases.update(test_cases_added)

        # 保存到target_path
        target_test_cases_path = test_cases_path.replace(base_path, target_path)
        os.makedirs(os.path.dirname(target_test_cases_path), exist_ok=True)
        with open(target_test_cases_path, 'w', encoding='utf-8') as f:
            json.dump(full_test_cases, f, indent=4)
    else:
        print(f"{method} {model} test_cases.json not found, skipping...")

    # 处理logs.json文件
    logs_path = os.path.join(parent_path, 'test_cases', 'logs.json')
    logs_path_added = logs_path.replace(base_path, base_path_added)

    if os.path.exists(logs_path) and os.path.exists(logs_path_added):
        with open(logs_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        with open(logs_path_added, 'r', encoding='utf-8') as f:
            logs_added = json.load(f)

        full_logs = {k: v for k, v in logs.items() if k in in_70_key}
        full_logs.update(logs_added)

        # 保存到target_path
        target_logs_path = logs_path.replace(base_path, target_path)
        os.makedirs(os.path.dirname(target_logs_path), exist_ok=True)
        with open(target_logs_path, 'w', encoding='utf-8') as f:
            json.dump(full_logs, f, indent=4)
    else:
        print(f"{method} {model} logs.json not found, skipping...")

    # print(f"Done processing {method} {model} test cases")


def process_completions(method, model, parent_path, defender_name='default'):
    # 处理completions.json文件
    if defender_name == 'default':
        completions_path = os.path.join(parent_path, 'completions', f'{model}.json')
    else:
        completions_path = os.path.join(parent_path, 'defense_completions', defender_name, f'{model}.json')

    completions_path_added = completions_path.replace(base_path, base_path_added)
    completions = {}
    completions_added = {}

    if not os.path.exists(completions_path):
        print(f"Completions file {completions_path} not found, skipping...")
        return None, None
    if not os.path.exists(completions_path_added):
        print(f"Completions file added {completions_path_added} not found, skipping...")
        return None, None

    with open(completions_path, 'r', encoding='utf-8') as f:
        completions = json.load(f)
    with open(completions_path_added, 'r', encoding='utf-8') as f:
        completions_added = json.load(f)

    full_completions = {k: v for k, v in completions.items() if k in in_70_key}
    full_completions.update(completions_added)

    # 保存到target_path
    target_completions_path = completions_path.replace(base_path, target_path)
    os.makedirs(os.path.dirname(target_completions_path), exist_ok=True)
    # print("="*500)
    # print(target_completions_path)
    with open(target_completions_path, 'w', encoding='utf-8') as f:
        json.dump(full_completions, f, indent=4)

    # print(f"Done processing {method} {model} completions")
        

def process_results(method, model, parent_path, defender_name='default', classifier_name='default'):
    # 处理results.json文件
    if defender_name == 'default' and classifier_name == 'default':
        results_path = os.path.join(parent_path, f'results', f'{model}.json')
    elif defender_name == 'default' and classifier_name != 'default':
        results_path = os.path.join(parent_path, f'results_{classifier_name}', f'{model}.json')
    elif defender_name != 'default' and classifier_name == 'default':
        results_path = os.path.join(parent_path, f'defense_results', defender_name, f'{model}.json')
    else:
        results_path = os.path.join(parent_path, f'defense_results_{classifier_name}', defender_name, f'{model}.json')

    results_path_added = results_path.replace(base_path, base_path_added)

    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found, skipping...")
        return None, None
    if not os.path.exists(results_path_added):
        print(f"Results file added {results_path_added} not found, skipping...")
        return None, None

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(results_path_added, 'r', encoding='utf-8') as f:
        results_added = json.load(f)

    full_results = {k: v for k, v in results.items() if k in in_70_key}
    full_results.update(results_added)

    # 保存到target_path
    target_results_path = results_path.replace(base_path, target_path)
    os.makedirs(os.path.dirname(target_results_path), exist_ok=True)
    with open(target_results_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=4)

    # print(f"Done processing {method} {model} results")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_name', type=str, default='default', help='Classifier name')
    parser.add_argument('--defender', type=str, default='default', help='Defender name')
    args = parser.parse_args()

    # qwen2_5_quantized_series = "qwen2_5_0_5b_instruct_gptq_int4,qwen2_5_1_5b_instruct_gptq_int4,qwen2_5_3b_instruct_gptq_int4,qwen2_5_7b_instruct_gptq_int4,qwen2_5_0_5b_instruct_gptq_int8,qwen2_5_1_5b_instruct_gptq_int8,qwen2_5_3b_instruct_gptq_int8,qwen2_5_7b_instruct_gptq_int8,qwen2_5_0_5b_instruct_awq,qwen2_5_1_5b_instruct_awq,qwen2_5_3b_instruct_awq,qwen2_5_7b_instruct_awq"
    # qwen3_series = "qwen3_0_6b,qwen3_1_7b,qwen3_4b"
    # qwen2_5_quantized_series = qwen2_5_quantized_series.split(',')  # 转换成列表
    # qwen3_series = qwen3_series.split(',')  # 转换成列表
    # model_list = qwen2_5_quantized_series + qwen3_series  # 添加到模型列表中
    
    for method in method_list:
        for model in model_list:
            parent_path = os.path.join(base_path, method, model)
            if method == 'DirectRequest':
                parent_path = os.path.join(base_path, method, 'default')
            elif method == 'HumanJailbreaks':
                parent_path = os.path.join(base_path, method, 'random_subset_5')
            elif method == 'PAP':
                parent_path = os.path.join(base_path, method, 'top_5')

            # 依次处理test_cases、completions和results
            process_test_cases(method, model, parent_path)
            process_completions(method, model, parent_path, defender_name=args.defender)
            process_results(method, model, parent_path, defender_name=args.defender, classifier_name=args.classifier_name)

            print("Done processing {} {}".format(method, model))




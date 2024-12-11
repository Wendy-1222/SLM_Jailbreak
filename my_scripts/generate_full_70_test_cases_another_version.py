import os
import json
from mothod_and_method_list import method_list, model_list

base_path = '/data2/zwh/HarmBench/results/'
base_path_added = '/data2/zwh/HarmBench/results_added_40/'
target_path = '/data2/zwh/HarmBench/results_full_70/'

in_70_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 23, 25, 26, 28, 31, 32, 38, 39, 41, 42,
              47]
in_70_key = [f'advbench_subset_{i}' for i in in_70_list]


def process_directory(source, added, target, filter_keys=None):
    if os.path.exists(source):
        os.makedirs(target, exist_ok=True)
        for subdir in os.listdir(source):
            if not filter_keys or subdir in filter_keys:
                os.system(f"cp -r {os.path.join(source, subdir)} {target}")
    if os.path.exists(added):
        os.makedirs(target, exist_ok=True)
        for subdir in os.listdir(added):
            os.system(f"cp -r {os.path.join(added, subdir)} {target}")


def process_json(source, added, target, filter_keys=None):
    if not os.path.exists(source) or not os.path.exists(added):
        return
    with open(source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(added, 'r', encoding='utf-8') as f:
        added_data = json.load(f)

    if filter_keys:
        data = {k: v for k, v in data.items() if k in filter_keys}
    data.update(added_data)

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def process_files(method, model, parent_path):
    paths = {
        "test_cases": ('test_cases', 'test_cases.json', 'logs.json'),
        "completions": ('completions', f'{model}.json'),
        "results": ('results', f'{model}.json')
    }

    for key, files in paths.items():
        dir_name = files[0]
        if key == "test_cases":
            process_directory(
                os.path.join(parent_path, dir_name, 'test_cases_individual_behaviors'),
                os.path.join(parent_path.replace(base_path, base_path_added), dir_name,
                             'test_cases_individual_behaviors'),
                os.path.join(parent_path.replace(base_path, target_path), dir_name, 'test_cases_individual_behaviors'),
                in_70_key
            )

        for file in files[1:]:
            process_json(
                os.path.join(parent_path, dir_name, file),
                os.path.join(parent_path.replace(base_path, base_path_added), dir_name, file),
                os.path.join(parent_path.replace(base_path, target_path), dir_name, file),
                in_70_key if "test_cases" in file or "logs" in file else None
            )


for method in method_list:
    for model in model_list:
        if method == 'DirectRequest':
            parent_path = os.path.join(base_path, method, 'default')
        elif method == 'HumanJailbreaks':
            parent_path = os.path.join(base_path, method, 'random_subset_5')
        elif method == 'PAP':
            parent_path = os.path.join(base_path, method, 'top_5')
        else:
            parent_path = os.path.join(base_path, method, model, 'results')

        process_files(method, model, parent_path)
        print(f"Done processing {method} {model}")

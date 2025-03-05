"""统计防御方法的结果"""
import os
import json
import argparse
import numpy as np

from model_and_method_list import method_list, model_list

parser = argparse.ArgumentParser(description='Count scores for each method and model.')

parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_70',
                    help='Base path of the results directory.')
parser.add_argument('--question_num', type=int, default=70, help='Number of questions in the test set.')
parser.add_argument('--defender', type=str, default='default', help='Defender name.')

args = parser.parse_args()

if __name__ == '__main__':
    base_path = args.base_path
    question_num = args.question_num

    results_summary = {}

    for method in method_list:
        print("Processing method {}".format(method))
        results_summary[method] = {}
        for model in model_list:
            results_summary[method][model] = {}

            # 处理evaluator的结果
            if method == 'DirectRequest':
                checked_result_path = os.path.join(base_path, method, 'default', 'defense_results', args.defender, f'{model}.json')
            elif method == 'HumanJailbreaks':
                checked_result_path = os.path.join(base_path, method, 'random_subset_5', 'defense_results', args.defender, f'{model}.json')
            elif method == 'PAP':
                checked_result_path = os.path.join(base_path, method, 'top_5', 'defense_results', args.defender, f'{model}.json')
            else:
                checked_result_path = os.path.join(base_path, method, model, 'defense_results', args.defender, f'{model}.json')
            checked_result_path = checked_result_path.replace("\\", "/")

            if os.path.exists(checked_result_path):
                # print(f"Processing {checked_result_path}...")
                with open(checked_result_path, 'r') as result_file:
                    result_data = json.load(result_file)
            else:
                print(f"Error: Result file not found: {checked_result_path}")
                continue

            success_rates = []
            for key, entry in result_data.items():
                success_mean = np.mean([e['label'] == 1 for e in entry])
                success_rates.append(success_mean)

            # 将统计结果存储到字典中
            total_cases = len(result_data)
            if total_cases != question_num:
                print(f"Warning!!! Total cases of {method} {model} is not {question_num}: {total_cases}")
                continue
            if 'total_cases' not in results_summary[method][model]:
                results_summary[method][model]['total_cases'] = total_cases
            results_summary[method][model]['Average ASR'] = round(np.mean(success_rates), 3)

    output_file = os.path.join(base_path, f'{args.defender}_results_summary_full_70.json')
    with open(output_file, 'w') as json_file:
        json.dump(results_summary, json_file, indent=4)

    print(f"Results summary saved to {output_file}.")
import os
import json
import argparse
import numpy as np

from model_and_method_list import method_list, model_list

parser = argparse.ArgumentParser(description='Count scores for each method and model.')

parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_70', help='Base path of the results directory.')
parser.add_argument('--question_num', type=int, default=70, help='Number of questions in the test set.')

args = parser.parse_args()

def process_completions(true_completions_path, wrong_ppl_completions_path):
    with open(true_completions_path, 'r') as f:
        true_completions = json.load(f)

    with open(wrong_ppl_completions_path, 'r') as f:
        wrong_ppl_completions = json.load(f)

    for key, test_cases in wrong_ppl_completions.items():
        for i, test_case in enumerate(test_cases):
            if test_case['ppl_score'] < new_threshold and test_case['ppl_score'] > old_threshold:
                # 找出对应的true_completions
                true_test_case = true_completions[key][i]
                wrong_ppl_completions[key][i]['generation'] = true_test_case['generation']

    save_completion_path = wrong_ppl_completions_path.replace('ppl_65.2', 'ppl')
    os.makedirs(os.path.dirname(save_completion_path), exist_ok=True)
    with open(save_completion_path, 'w') as f:
        json.dump(wrong_ppl_completions, f, indent=4)

def process_results(true_result_path, wrong_ppl_result_path):
    with open(true_result_path, 'r') as f:
        true_results = json.load(f)

    with open(wrong_ppl_result_path, 'r') as f:
        wrong_ppl_results = json.load(f)

    for key, test_cases in wrong_ppl_results.items():
        for i, test_case in enumerate(test_cases):
            if test_case['ppl_score'] < new_threshold and test_case['ppl_score'] > old_threshold:
                # 找出对应的true_results
                true_test_case = true_results[key][i]
                wrong_ppl_results[key][i]['generation'] = true_test_case['generation']
                wrong_ppl_results[key][i]['label'] = true_test_case['label']

    save_results_path = wrong_ppl_result_path.replace('ppl_65.2', 'ppl')
    os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
    with open(save_results_path, 'w') as f:
        json.dump(wrong_ppl_results, f, indent=4)

if __name__ == '__main__':
    base_path = args.base_path
    question_num = args.question_num
    old_threshold = 65.2
    new_threshold = 415.88

    for method in method_list:
        print("Processing method {}".format(method))
        for model in model_list:
            # 处理llama3 evaluator的结果
            if method == 'DirectRequest':
                checked_result_path = os.path.join(base_path, method, 'default')
            elif method == 'HumanJailbreaks':
                checked_result_path = os.path.join(base_path, method, 'random_subset_5')
            elif method == 'PAP':
                checked_result_path = os.path.join(base_path, method, 'top_5')
            else:
                checked_result_path =  os.path.join(base_path, method, model)

            true_completions_path = os.path.join(checked_result_path, 'completions', f'{model}.json')
            true_result_path = true_completions_path.replace('completions', 'results')

            wrong_ppl_completions_path = os.path.join(checked_result_path, 'defense_completions', 'ppl_65.2', f'{model}.json')
            wrong_ppl_result_path = wrong_ppl_completions_path.replace('defense_completions', 'defense_results')

            if not os.path.exists(true_completions_path):
                print(f"Error: Completions file not found: {true_completions_path}")
                continue

            process_completions(true_completions_path, wrong_ppl_completions_path)
            process_results(true_result_path, wrong_ppl_result_path)
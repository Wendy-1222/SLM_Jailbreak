import os
import argparse
import json

from model_and_method_list import method_list, model_list

parser = argparse.ArgumentParser(description='Count scores for each method and model.')

parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_70', help='Base path of the results directory.')
parser.add_argument('--question_num', type=int, default=70, help='Number of questions in the test set.')

args = parser.parse_args()

if __name__ == '__main__':
    base_path = args.base_path
    question_num = args.question_num

    defender_list = ['ppl', 'retokenization', 'self-reminder']
    for defender in defender_list:
        print("="*100, defender)
        for method in method_list:
            print("Processing method {}".format(method))
            for model in model_list:
                if method == 'DirectRequest':
                    checked_result_path = os.path.join(base_path, method, 'default')
                elif method == 'HumanJailbreaks':
                    checked_result_path = os.path.join(base_path, method, 'random_subset_5')
                elif method == 'PAP':
                    checked_result_path = os.path.join(base_path, method, 'top_5')
                else:
                    checked_result_path =  os.path.join(base_path, method, model)

                checked_completions_path = os.path.join(checked_result_path, 'defense_completions', defender, f'{model}.json')
                checked_result_path = checked_completions_path.replace('defense_completions', 'defense_results')
                checked_result_path = checked_result_path.replace('\\', '/')
                checked_result_path = checked_result_path.replace("\\", "/")

                if os.path.exists(checked_completions_path):
                    with open(checked_completions_path, 'r') as completion_file:
                        completion_data = json.load(completion_file)
                        if len(completion_data) != args.question_num:
                            print(f"Warning!!! Total cases of {method} {model} is not {question_num}: {len(completion_data)}")
                else:
                    print(f"Error: Completions file not found: {checked_completions_path}")
                    continue

                if os.path.exists(checked_result_path):
                    with open(checked_result_path, 'r') as result_file:
                        result_data = json.load(result_file)
                        if len(result_data) != args.question_num:
                            print(f"Warning!!! Total cases of {method} {model} is not {question_num}: {len(result_data)}")
                else:
                    print(f"Error: Result file not found: {checked_result_path}")
                    continue




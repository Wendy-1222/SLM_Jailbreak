import os
import json
import argparse

from model_and_method_list import method_list, model_list

parser = argparse.ArgumentParser(description='Count scores for each method and model.')

parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_50', help='Base path of the results directory.')
parser.add_argument('--question_num', type=int, default=50, help='Number of questions in the test set.')

args = parser.parse_args()

metric_list = ['repetition_rate', 'distinct_2', 'entropy_2', 'average_sentence_length', 'lexical_diversity',
               'word_nums', 'sentence_nums', 'self_bleu', 'perplexity', 'readability_score', 'coherence_score']

if __name__ == '__main__':
    base_path = args.base_path
    question_num = args.question_num

    results_summary = {}

    for method in method_list:
        print("Processing method {}".format(method))
        results_summary[method] = {}
        for model in model_list:
            results_summary[method][model] = {}
           
            # 得到结果的路径
            if method == 'DirectRequest':
                base_result_path = os.path.join(base_path, method, 'default')
            elif method == 'HumanJailbreaks':
                base_result_path = os.path.join(base_path, method, 'random_subset_5')
            elif method == 'PAP':
                base_result_path = os.path.join(base_path, method, 'top_5')
            else:
                base_result_path = os.path.join(base_path, method, model)
            
            # 处理不同metric的结果
            for metric in metric_list:
                results_summary[method][model][metric] = {}

                result_path = os.path.join(base_result_path, 'metrics_jailbreak', metric, f'{model}.json')

                if os.path.exists(result_path):
                    print(f"Processing {result_path}...")
                else:
                    print(f"Error: Result file not found: {result_path}")
                    results_summary[method][model][metric]['effective_runs'] = 0
                    results_summary[method][model][metric]['average_value'] = -1
                    continue

                with open(result_path, 'r') as result_file:
                    result_data = json.load(result_file)

                all_run_scores = 0
                effective_runs = 0
                for run_id, run_data in result_data.items():
                    # 首先对run_data包含多个test_case的情况计算平均值
                    if len(run_data) > 1:
                        single_run_scores = [item[metric] for item in run_data if item[metric] != -1]
                        average_score_single_run = sum(single_run_scores) / len(single_run_scores) if single_run_scores else -1
                    else:
                        average_score_single_run = run_data[0][metric]
                    if average_score_single_run == -1:
                        continue
                    all_run_scores += average_score_single_run
                    effective_runs += 1

                average_score = round(all_run_scores / effective_runs, 4) if effective_runs > 0 else -1
                
                # 将统计结果存储到字典中
                results_summary[method][model][metric]['effective_runs'] = effective_runs
                results_summary[method][model][metric]['average_value'] = average_score
    
    output_file = os.path.join(base_path, 'jailbreak_results_summary_different_metrics.json')
    with open(output_file, 'w') as json_file:
        json.dump(results_summary, json_file, indent=4)

    print(f"Results summary saved to {output_file}.")
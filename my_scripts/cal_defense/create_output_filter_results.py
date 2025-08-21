import os
import json
import argparse
import numpy as np
from copy import deepcopy

from model_and_method_list import method_list, model_list

parser = argparse.ArgumentParser(description='Create output filter results according to classifier results.')

parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_70',
                    help='Base path of the results directory.')
parser.add_argument('--question_num', type=int, default=70, help='Number of questions in the test set.')
parser.add_argument('--defender', type=str, default='default', help='Defender name.')

args = parser.parse_args()

method_list = ["AutoDAN"]
model_list = ["qwen2_5_1_5b_instruct_gptq_int4", "qwen2_5_3b_instruct_gptq_int4", "qwen2_5_7b_instruct_gptq_int4", "qwen2_5_0_5b_instruct_gptq_int8", "qwen2_5_1_5b_instruct_gptq_int8", "qwen2_5_3b_instruct_gptq_int8", "qwen2_5_7b_instruct_gptq_int8"]

if __name__ == '__main__':
    base_path = args.base_path
    question_num = args.question_num
    
    for method in method_list:
        print("Processing method {}".format(method))
        for model in model_list:
            # 得到基础的结果和output_filter的结果
            if method == 'DirectRequest':
                output_filter_result_path = os.path.join(base_path, method, 'default', f'results_{args.defender}', f'{model}.json')
            elif method == 'HumanJailbreaks':
                output_filter_result_path = os.path.join(base_path, method, 'random_subset_5', f'results_{args.defender}', f'{model}.json')
            elif method == 'PAP':
                output_filter_result_path = os.path.join(base_path, method, 'top_5', f'results_{args.defender}', f'{model}.json')
            else:
                output_filter_result_path = os.path.join(base_path, method, model, f'results_{args.defender}', f'{model}.json')
            output_filter_result_path = output_filter_result_path.replace("\\", "/")

            if os.path.exists(output_filter_result_path):
                # print(f"Processing {output_filter_result_path}...")
                with open(output_filter_result_path, 'r') as result_file:
                    output_filter_result_data = json.load(result_file)
            else:
                print(f"Error: Result file not found: {output_filter_result_path}")
                continue
            
            checked_result_path = output_filter_result_path.replace(f"/results_{args.defender}/", "/results/")
            with open(checked_result_path, 'r') as result_file:
                result_data = json.load(result_file)
                
            # 创建新的字典存储结果
            new_result_data = {}

            # 遍历 result_data 和 output_filter_data 中的每一项
            for (id, result_items), (id_1, output_items) in zip(result_data.items(), output_filter_result_data.items()):
                new_data_list = []

                for result_item, output_item in zip(result_items, output_items):
                    new_item = {}
                    
                    assert result_item['test_case'] == output_item['test_case'], "Test cases do not match between result and output filter data."

                    new_item['test_case'] = result_item['test_case']
                    new_item['generation'] = result_item['generation']
                    
                    # 对比并保留 result_data 中的 label，排除 output_filter_data 中的 label
                    new_item['output_filter_label'] = output_item.get('label', False)
                    new_item['llama_2_cls_label'] = result_item.get('label', False)
                    new_item['label'] = new_item['llama_2_cls_label'] and not new_item['output_filter_label']
                    
                    # 确保是数字1或0，而不是布尔值
                    new_item['label'] = int(new_item['label'])
                    
                    # 保留 output_filter_data 中的其他字段（如 raw_output 等）
                    for key, value in output_item.items():
                        if key not in ['test_case', 'generation', 'label']:
                            new_item[key] = value
                    
                    # 将新字典添加到当前 id 下的列表
                    new_data_list.append(new_item)
                
                # 将处理后的数据添加到新字典中
                new_result_data[id] = new_data_list

            save_path = output_filter_result_path.replace(f"/results_{args.defender}/", f"/defense_results/{args.defender}_output_filter/")     
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(save_path, 'w') as f:
                json.dump(new_result_data, f, indent=4)

            print(f"Processed method {method} for model {model}.")
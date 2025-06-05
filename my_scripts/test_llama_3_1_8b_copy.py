"""
使用llama 3.1 8B根据OpenAI policy对completions进行评分
"""

import torch
import os
import json
import pandas as pd
import time
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_and_method_list import model_list, method_list
from judge_prompt import get_evaluator_system_prompt_for_judge, get_evaluator_prompt_for_judge, process_output_judge_score

torch.cuda.empty_cache()

model_dir = "/data2/SLMs/llama3.1/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型
llama3_2_8B_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    # device_map='auto',
    trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
llama3_2_8B_model = llama3_2_8B_model.to(device)


# def Ask(text):
def Ask(messages):
    # inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    output = llama3_2_8B_model.generate(input_ids=input_ids, max_new_tokens=2000, pad_token_id=0, do_sample=False)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def get_evaluator_response(goal, jailbreak_prompt, response):
    evaluator_system_prompt = get_evaluator_system_prompt_for_judge(goal)
    evaluate_template = "GOAL: {goal}\n{instruction}\nASSISTANT'S RESPONSE: {response}"
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluate_template.format(goal=goal, instruction=jailbreak_prompt, response=response)}
    ]
    raw_output = Ask(messages)
    # print(f"raw_output of judging socre: {raw_output}")

    score, explanation = process_output_judge_score(raw_output)
    return raw_output, score, explanation

def get_question(case_id):
    if case_id.startswith("advbench_subset"):
        data_path = '../data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv'
    elif case_id.startswith("adjusted_advbench_added_"):
        data_path = '../data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv'

    case_id_value = int(case_id.split('_')[-1])

    # 获取case_id对应的question. 结构为Behavior,Category,Tags,ContextString,BehaviorID
    df = pd.read_csv(data_path)
    question = df[df['BehaviorID'] == case_id_value]['Behavior']

    return question
        

# 处理一个project的所有run，返回一个新的df
def process_project(completions_path):
    # 读取json文件
    with open(completions_path, 'r') as f:
        completions_data = json.load(f)

    results = {}
    for run_id, run_data in completions_data.items():   
        question = get_question(run_id)

        raw_outputs = []
        scores = []
        explanations = []
        for case in run_data:
            raw_output, score, explanation = get_evaluator_response(question, case["test_case"], case["generation"])
            raw_outputs.append(raw_output)
            scores.append(score)
            explanations.append(explanation)

        # 为每个generation添加raw_output、score和explanation
        processed_data = []
        for case, raw_output, score, explanation in zip(run_data, raw_outputs, scores, explanations):
            processed_case = case.copy()
            processed_case["raw_output"] = raw_output
            processed_case["score"] = score
            processed_case["explanation"] = explanation
            processed_data.append(processed_case)

        results[run_id] = processed_data
        print("Done processing run_id: {}".format(run_id))

    return results

base_path = '/data2/zwh/HarmBench/results/'
results_summary = {}

method_list = ["AutoPrompt", "PEZ", "UAT", "GBDA"]
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
        result_path = completions_path.replace("/completions/", "/results_llama3_1_8b/")
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





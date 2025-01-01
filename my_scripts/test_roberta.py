"""
使用roberta模型对所有的completions进行预测, 输出confidence和label
"""
import pandas as pd
import wandb
from tqdm import tqdm
import json
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import time
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

from judge_prompt import get_evaluator_system_prompt_for_judge, get_evaluator_prompt_for_judge, process_output_judge_score, extract_label_and_category
from model_and_method_list import model_list, method_list

class Predictor:
    def __init__(self, path):
        self.path = path

    def predict(self, sequences):
        raise NotImplementedError("Predictor must implement predict method.")

class RoBERTaPredictor(Predictor):
    def __init__(self, path, device='cuda'):
        super().__init__(path)
        self.device = device
        self.model = RobertaForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(path)

    def predict(self, sequences):
        inputs = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidences, predicted_classes = torch.max(predictions, dim=1)
        return confidences.cpu().tolist(), predicted_classes.cpu().tolist()
    
class Llama3_8b_Predictor(Predictor):
    def __init__(self, path, device='cuda'):
        super().__init__(path)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def get_model_response(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=2000, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    def predict(self, goals_list, response_list):
        raw_output_list = []
        score_list = []
        explanation_list = []
        for goal, response in zip(goals_list, response_list):
            evaluator_system_prompt = get_evaluator_system_prompt_for_judge(goal)
            evaluate_template = "GOAL: {goal}\nASSISTANT'S RESPONSE: {response}"
            messages = [
                {"role": "system", "content": evaluator_system_prompt},
                {"role": "user", "content": evaluate_template.format(goal=goal, response=response)}
            ]
            raw_output = self.get_model_response(messages)
            # print(f"raw_output: {raw_output}")
            score, explanation = process_output_judge_score(raw_output)
            raw_output_list.append(raw_output)
            score_list.append(score)
            explanation_list.append(explanation)

        return raw_output_list, score_list, explanation_list


class Llama3_guard_Predictor(Predictor):
    def __init__(self, path, device='cuda'):
        super().__init__(path)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def get_model_response(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=2000, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def predict(self, goals_list, response_list):
        raw_output_list = []
        label_list = []
        category_list = []
        for goal, response in zip(goals_list, response_list):
            messages = [
                {"role": "user", "content": goal},
                {"role": "assistant", "content": response},
            ]
            raw_output = self.get_model_response(messages)
            # print(f"raw_output: {raw_output}")
            label, category = extract_label_and_category(raw_output)
            raw_output_list.append(raw_output)
            label_list.append(label)
            category_list.append(category)

        return raw_output_list, label_list, category_list


def get_goal_from_run_id(source_id):
    # 加载数据
    full_50_data = pd.read_csv("../data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv")
    full_40_data = pd.read_csv("../data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv")

    # 根据 source_id 查询数据
    if source_id.startswith('advbench_subset'):
        row = full_50_data[full_50_data['BehaviorID'] == source_id]
    else:
        row = full_40_data[full_40_data['BehaviorID'] == source_id]

    # 检查是否找到匹配的行
    if row.empty:
        raise ValueError(f"Error: {source_id} not found in datasets.")  # 更清晰的错误信息

    # 返回目标值，统一为字符串
    return row.iloc[0]['Behavior']  # 直接提取第一个匹配值
    

# 处理一个project的所有run，返回一个新的df
def process_project(completions_path, classifier_name='roberta'):
    # 读取json文件
    with open(completions_path, 'r') as f:
        completions_data = json.load(f)

    results = {}
    if classifier_name == 'roberta':
        for run_id, run_data in completions_data.items():
            response_list = [case["generation"] for case in run_data]
            confidences, predict_results = roberta_model.predict(response_list)
            
            # 为每个generation添加confidence和label
            processed_data = []
            for case, confidence, label in zip(run_data, confidences, predict_results):
                processed_case = case.copy()
                processed_case["confidence"] = confidence
                processed_case["label"] = label
                processed_data.append(processed_case)

            results[run_id] = processed_data
    elif classifier_name == 'llama3_2_8b':
        for run_id, run_data in completions_data.items():
            goals_list = [get_goal_from_run_id(run_id)] * len(run_data)
            jailbreak_prompts_list = [case["test_case"] for case in run_data]
            response_list = [case["generation"] for case in run_data]
            raw_output_list, score_list, explanation_list = llama3_2_8b_model.predict(goals_list, response_list)
            
            # 为每个generation添加score和explanation
            processed_data = []
            for case, raw_output, score, explanation in zip(run_data, raw_output_list, score_list, explanation_list):
                if score == -1:
                    print(f"Error extract score in processing {completions_path} {run_id}")
                processed_case = case.copy()
                processed_case["raw_output"] = raw_output
                processed_case["score"] = score
                processed_case["explanation"] = explanation
                processed_data.append(processed_case)

            results[run_id] = processed_data
    elif classifier_name == 'llama3_guard':
        for run_id, run_data in completions_data.items():
            goals_list = [get_goal_from_run_id(run_id)] * len(run_data)
            # jailbreak_prompts_list = [case["test_case"] for case in run_data]
            response_list = [case["generation"] for case in run_data]
            raw_output_list, label_list, category_list = llama3_guard_model.predict(goals_list, response_list)
            
            # 为每个generation添加label和category
            processed_data = []
            for case, raw_output, label, category in zip(run_data, raw_output_list, label_list, category_list):
                processed_case = case.copy()
                processed_case["raw_output"] = raw_output
                processed_case["label"] = label
                processed_case["category"] = category
                processed_data.append(processed_case)

            results[run_id] = processed_data

    return results


parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results/')
parser.add_argument('--classifier_name', type=str, default='llama3_2_8b', choices=['roberta', 'llama3_2_8b', 'llama3_guard'])
args = parser.parse_args()

results_summary = {}

if args.classifier_name == 'roberta':
    roberta_model = RoBERTaPredictor('/data2/zwh/models/RoBerta', device='cuda:0')
elif args.classifier_name == 'llama3_2_8b':
    llama3_2_8b_model = Llama3_8b_Predictor('/data2/llama3.1/Llama-3.1-8B-Instruct', device='cuda:0')
elif args.classifier_name == 'llama3_guard':
    llama3_guard_model = Llama3_guard_Predictor('/data2/llama3.1/Llama-Guard-3-8B', device='cuda:0')
print("Done loading model")

for method in method_list:
    print("Processing method {}".format(method))
    results_summary[method] = {}
    for model in model_list:
        results_summary[method][model] = {}
        
        # 获得completions的路径
        if method == 'DirectRequest':
            completions_path = os.path.join(args.base_path, method, 'default', 'completions', f'{model}.json')
        elif method == 'HumanJailbreaks':
            completions_path = os.path.join(args.base_path, method, 'random_subset_5', 'completions', f'{model}.json')
        elif method == 'PAP':
            completions_path = os.path.join(args.base_path, method, 'top_5', 'completions', f'{model}.json')
        else:
            completions_path = os.path.join(args.base_path, method, model, 'completions', f'{model}.json')
        completions_path = completions_path.replace("\\", "/")

        if not os.path.exists(completions_path):
            print(f"File {completions_path} not found, skipping...")
            continue

        result_path = completions_path.replace("/completions/", f"/results_{args.classifier_name}/")
        # 获取父目录路径
        parent_directory = os.path.dirname(result_path)

        # 检查文件是否存在
        if os.path.exists(result_path):
            print(f"File {result_path} already exists, skipping...")
            continue
        else:
            # 创建父目录（如果不存在）
            os.makedirs(parent_directory, exist_ok=True)
            print(f"Created directory {parent_directory}")

        # 使用分类器进行预测
        project_results = process_project(completions_path, args.classifier_name)

        # 保存实验结果
        with open(result_path, 'w') as f:
            json.dump(project_results, f, indent=4)

        print(f"Processed results saved to {result_path}")

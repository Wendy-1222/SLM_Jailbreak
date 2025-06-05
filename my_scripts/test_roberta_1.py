"""
使用roberta模型对所有的completions进行预测, 输出confidence和label
"""
import pandas as pd
from tqdm import tqdm
import json
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

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

# 处理一个project的所有run，返回一个新的df
def process_project(completions_path):
    # 读取json文件
    with open(completions_path, 'r') as f:
        completions_data = json.load(f)

    results = {}
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

    return results

base_path = '/data2/zwh/HarmBench/results_full_50/'
results_summary = {}

roberta_model = RoBERTaPredictor('/data2/zwh/models/RoBERTa', device='cuda:0')
print("Done loading model")

method_list = ['UAT']

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
        result_path = completions_path.replace("/completions/", "/results_roberta/")
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

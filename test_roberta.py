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


method_list = ['DirectRequest', 'HumanJailbreaks', 'PAP', 'PAIR', 'GCG', 'AutoPrompt', 'PEZ', 'GBDA', 'UAT']
print(method_list)
qwen_series = "qwen_1_8b_chat,qwen_7b_chat,qwen1_5_0_5b_chat,qwen1_5_1_8b_chat,qwen1_5_4b_chat,qwen1_5_7b_chat,qwen2_0_5b_instruct,qwen2_1_5b_instruct,qwen2_7b_instruct,qwen2_5_0_5b_instruct,qwen2_5_1_5b_instruct,qwen2_5_3b_instruct,qwen2_5_7b_instruct"
pythia_series = "pythia_14m,pythia_31m,pythia_70m,pythia_160m,pythia_410m,pythia_1b,pythia_1_4b,pythia_2_8b,pythia_6_9b"
phi_series = "phi_1_5,phi_2,phi_3_mini_4k_instruct,phi_3_mini_128k_instruct,phi_3_small_8k_instruct,phi_3_small_128k_instruct,phi_3_5_mini_instruct"
stablelm_series = "stablelm-2-zephyr-1_6b,stablelm-2-1_6b-chat,stablelm-zephyr-3b"
tiny_llama_series = "tinyllama-1.1B-chat-v0.1,tinyllama-1.1B-chat-v0.2,tinyllama-1.1B-chat-v0.3,tinyllama-1.1B-chat-v0.4,tinyllama-1.1B-chat-v0.5,tinyllama-1.1B-chat-v0.6,tinyllama-1.1B-chat-v1.0"
mobile_llama_series="mobilellama-1.4B-chat,mobilellama-2.7B-chat"
mobi_llama_series="mobillama-0.5B-chat,mobillama-1B-chat"
gemma_series="gemma-2b-it,gemma-7b-it,gemma-1.1-2b-it,gemma-1.1-7b-it,gemma-2-2b-it,recurrentgemma-2b-it"
minicpm_series="minicpm-1B-sft-bf16,minicpm-S-1B-sft,minicpm-2B-sft-bf16,minicpm-2B-sft-fp32,minicpm-2B-sft-int4,minicpm-2B-dpo-bf16,minicpm-2B-dpo-fp16,minicpm-2B-dpo-fp32,minicpm-2B-dpo-int4,minicpm-2B-128k,minicpm3-4B"
h2o_danube_series="h2o-danube-1.8b-sft,h2o-danube-1.8b-chat,h2o-danube2-1.8b-sft,h2o-danube2-1.8b-chat,h2o-danube3-500m-chat,h2o-danube3-4b-chat"
fox_series = "fox-1-1.6B-Instruct-v0.1"
smollm_series = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct"
dclm_series = "DCLM-1B-IT"
dolly_series = "dolly-v1-6b,dolly-v2-3b,dolly-v2-7b"
olmo_series = "OLMo-7B-SFT-hf,OLMo-7B-Instruct-hf"

# 转换成列表
qwen_series = qwen_series.split(',')
pythia_series = pythia_series.split(',')
phi_series = phi_series.split(',')
stablelm_series = stablelm_series.split(',')
tiny_llama_series = tiny_llama_series.split(',')
mobile_llama_series = mobile_llama_series.split(',')
mobi_llama_series = mobi_llama_series.split(',')
gemma_series = gemma_series.split(',')
minicpm_series = minicpm_series.split(',')
h2o_danube_series = h2o_danube_series.split(',')
fox_series = fox_series.split(',')
smollm_series = smollm_series.split(',')
dclm_series = dclm_series.split(',')
dolly_series = dolly_series.split(',')
olmo_series = olmo_series.split(',')
model_list = ['llama2_7b', 'vicuna_7b_v1_5']
model_list += qwen_series + pythia_series + phi_series + stablelm_series + tiny_llama_series + mobile_llama_series + mobi_llama_series + gemma_series + minicpm_series + h2o_danube_series + fox_series + smollm_series + dclm_series + dolly_series + olmo_series
print(model_list)

base_path = '/data2/zwh/HarmBench/results/'
results_summary = {}

roberta_model = RoBERTaPredictor('/data/zwh/models/RoBerta(GPTFuzz)', device='cuda:0')
print("Done loading model")

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
            completions_path =  os.path.join(base_path, method, model, 'results', f'{model}.json')
        completions_path = completions_path.replace("\\", "/")

        if not os.path.exists(completions_path):
            print(f"File {completions_path} not found, skipping...")
            continue

        project_results = process_project(completions_path)

        # 修改路径为 results 并更改文件名
        result_path = completions_path.replace("/completions/", "/results_roberta/")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(project_results, f, indent=4)

        print(f"Processed results saved to {result_path}")

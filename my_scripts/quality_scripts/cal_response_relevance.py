import argparse
import os
import sys
import json
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

from flow_judge import Vllm, Llamafile, Hf, EvalInput, FlowJudge
from flow_judge.metrics import list_all_metrics, RESPONSE_RELEVANCE_3POINT

# 动态添加项目根目录到 Python 路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

from my_scripts.model_and_method_list import model_list, method_list
from my_scripts.more_evaluator.judge_prompt import get_evaluator_system_prompt_for_relevance 



def cal_cosine_similarity(prompt, response):
    # 使用TF-IDF向量化器将文本转化为向量
    vectorizer = TfidfVectorizer()
    
    # 将 prompt 和 response 转换为TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([prompt, response])
    
    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    raw_output = cosine_sim[0][0]
    
    return raw_output, cosine_sim[0][0]


def cal_llm_relevance(prompt, response):
    evaluator_system_prompt = get_evaluator_system_prompt_for_relevance(prompt, response)
    evaluate_template = "QUESTION: {question}\nRESPONSE: {response}"
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluate_template.format(question=prompt, response=response)}
    ]

    # inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    output = llama3_2_8B_model.generate(input_ids=input_ids, max_new_tokens=128, pad_token_id=0, do_sample=False)
    prompt_len = input_ids.shape[-1]
    raw_output = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    # print(f"raw_output of judging socre: {raw_output}")

    if "1" in raw_output:
        score = 1
    elif "0" in raw_output:
        score = 0
    else:
        score = -1

    print(f"prompt: {prompt}")
    print(f"response: {response}")
    print(f"raw_output: {raw_output}")
    print(f"score: {score}\n")
    return raw_output, score

def flow_judge_relevance(prompt, response):
    eval_input = EvalInput(
        inputs=[
            {"query": prompt},
            {"context": ""},
        ],
        output={"response": response},
    )

    result = relevance_judge.evaluate(eval_input, save_results=False)
    print(result)
    return result.feedback, result.score  # 直接返回result似乎会报格式错误


# 处理每个模型的生成结果
def process_completions(completions_path, metric_name, metric_function):
    with open(completions_path, 'r') as f:
        completions_data = json.load(f)

    # 初始化指标结果字典
    metrics_results = {}

    # 记录所有 response 全为空的 run_id 数量
    empty_run_count = 0

    # 遍历每个 run_id 和对应的生成
    for run_id, run_data in completions_data.items():
        question = question_dict[run_id]
        # prompt_list = [case["test_case"] for case in run_data]  # 实际上不需要
        response_list = [case["generation"] for case in run_data]
        # print(len(response_list))

        # 检测该 run_id 的所有 response 是否全为空
        if all(not response.strip() for response in response_list):  # 如果所有 response 都为空
            empty_run_count += 1  # 统计空 run_id 的数量
            continue  # 跳过处理这个 run_id 的逻辑

        cal_results = []
        raw_outputs = []
        for response in response_list:
            if not response.strip():  # 如果文本为空 
                cal_results.append(-1)
            else:
                raw_output, score = metric_function(question, response)
                raw_outputs.append(raw_output)
                cal_results.append(round(score, 4))
        # print(cal_results)
        
        # 为每个 generation 添加计算结果
        processed_data = []
        for case, metric, raw_output in zip(run_data, cal_results, raw_outputs):
            case_copy = case.copy()
            case_copy["raw_output"] = raw_output
            case_copy[metric_name] = metric
            processed_data.append(case_copy)

        metrics_results[run_id] = processed_data
        # print(metrics_results[run_id])

    return metrics_results, empty_run_count


parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_70/')
parser.add_argument('--question_num', type=int, default=70, help='Number of questions in the test set.')
parser.add_argument('--metric', type=str, required=True, choices=[
    'cosine_similarity', 'llm_relevance', "flow_judge"
])
parser.add_argument('--model_name', type=str, default='/data2/SLMs/llama3.1/Llama-3.1-8B-Instruct', help='Model name.')
parser.add_argument('--question_file_path', type=str, default='/data2/zwh/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors_subset.csv', help='Path to the final_adjusted_advbench.csv file.')
parser.add_argument('--question_file_path_added', type=str, default='/data2/zwh/HarmBench/data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv', help='Path to the advbench_behaviors_subset_added.csv file.')

args = parser.parse_args()

# 加载模型
if args.metric == 'llm_relevance':
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llama3_2_8B_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama3_2_8B_model.to(device)

    print("Models loaded")
elif args.metric == 'flow_judge':
    model = Vllm()
    relevance_judge = FlowJudge(
        metric=RESPONSE_RELEVANCE_3POINT,
        model=model
    )

# 加载计算指标的函数
metric_functions = {
    "cosine_similarity": cal_cosine_similarity,
    "llm_relevance": cal_llm_relevance,
    "flow_judge": flow_judge_relevance,
}
args.metric_function = metric_functions[args.metric]

question_dict = {}
# 加载问题字典（从BehaviorID到Behavior的映射）
df1 = pd.read_csv(args.question_file_path)
question_dict = df1.set_index('BehaviorID')['Behavior'].to_dict()

df2 = pd.read_csv(args.question_file_path_added)
question_dict_added = df2.set_index('BehaviorID')['Behavior'].to_dict()
question_dict.update(question_dict_added)

assert len(question_dict) == 90, "The number of questions is not 90"

# 遍历越狱方法和模型
for method in method_list:
    print(f"Processing method: {method}")
    for model in model_list:
        # 获得results的路径
        if method == 'DirectRequest':
            completions_path = os.path.join(args.base_path, method, 'default', 'results', f'{model}.json')
        elif method == 'HumanJailbreaks':
            completions_path = os.path.join(args.base_path, method, 'random_subset_5', 'results', f'{model}.json')
        elif method == 'PAP':
            completions_path = os.path.join(args.base_path, method, 'top_5', 'results', f'{model}.json')
        else:
            completions_path = os.path.join(args.base_path, method, model, 'results', f'{model}.json')
        completions_path = completions_path.replace("\\", "/")

        if not os.path.exists(completions_path):
            print(f"File {completions_path} not found, skipping...")
            continue
        else:
            with open(completions_path, 'r') as f:
                temp_completions_data = json.load(f)
                completions_num = len(temp_completions_data)
                if completions_num != args.question_num:
                    print(f"Warning!!! Total cases of {method} {model} is not {args.question_num}: {completions_num}")
                    continue

        result_path = completions_path.replace("/results/", f"/metrics/{args.metric}/")
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
        
        metric_results, empty_run_count = process_completions(completions_path, args.metric, args.metric_function)

        # 保存实验结果
        with open(result_path, 'w') as f:
            json.dump(metric_results, f, indent=4)

        print(f"Processed results saved to {result_path}")

import argparse
import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textstat
from collections import Counter
from math import log2

from model_and_method_list import model_list, method_list

# 指标计算函数
def calculate_repetition_rate(text, n=2):
    """
    计算文本的重复率（Repetition Rate）。此处统计一个文本中重复出现的 n-gram（如2-gram或3-gram）占总 n-gram 的比例。
    参数:
        text: 输入文本 (字符串)
        n: n-gram 的大小
    返回:
        重复率 (浮点数)
    """
    words = text.split()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    repeated_ngrams = sum(count for count in ngram_counts.values() if count > 1)
    total_ngrams = len(ngrams)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0

def calculate_distinct_n(text, n=2):
    """
    计算文本的 Distinct-n 指标，即生成文本中独特 n-gram 的比例，以衡量生成文本的多样性
    参数:
        text: 输入文本 (字符串)
        n: n-gram 的大小
    返回:
        Distinct-n (浮点数)
    """
    words = text.split()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    unique_ngrams = set(ngrams)
    total_ngrams = len(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

def calculate_entropy_n(text, n=2):
    """
    计算文本的 Entropy-n, 即文本中 n-gram 的信息熵，用于衡量生成文本的词汇多样性和随机性。
    参数:
        text: 输入文本 (字符串)
        n: n-gram 的大小
    返回:
        Entropy-n (浮点数)
    """
    words = text.split()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    entropy = 0.0
    for count in ngram_counts.values():
        p = count / total_ngrams
        entropy -= p * log2(p)
    return entropy

def calculate_average_sentence_length(text):
    """
    计算文本的平均句长。
    参数:
        text: 输入文本 (字符串)
    返回:
        平均句长 (浮点数)
    """
    sentences = text.split(". ")
    sentences = [s.strip() for s in sentences if s.strip()]
    total_words = sum(len(sentence.split()) for sentence in sentences)
    return total_words / len(sentences) if sentences else 0.0

def calculate_sentence_nums(text):
    """
    计算文本的句子数量。
    参数:
        text: 输入文本 (字符串)
    返回:
        文本的句子数量
    """
    sentences = text.split(". ")
    return len(sentences)

def calculate_word_nums(text):
    """
    计算文本的单词数量。
    参数:
        text: 输入文本 (字符串)
    返回:
        文本的单词数量
    """
    words = text.split()
    return len(words)

def calculate_lexical_diversity(text):
    """
    计算文本的词汇多样性，即计算文本中独特单词占总单词比例来衡量。
    参数:
        text: 输入文本 (字符串)
    返回:
        词汇多样性 (浮点数)
    """
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0.0

def calculate_perplexity(text, model, tokenizer):
    """
    使用 GPT-2 模型计算文本的困惑度 (Perplexity)。
    参数:
        text: 输入文本 (字符串)
        model: GPT-2 模型
        tokenizer: GPT-2 分词器
    返回:
        困惑度 (浮点数)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

def calculate_readability_score(text):
    """
    计算文本的可读性得分。使用可读性测量公式（如 Flesch-Kincaid 或 Gunning Fog Index) 评估文本的可读性。
    参数:
        text: 输入文本 (字符串)
    返回:
        可读性得分 (Flesch Reading Ease)
    """
    return textstat.flesch_reading_ease(text)

def calculate_coherence_score(text, sentence_model):
    """
    计算文本的连贯性得分。通过话题模型（如 LDA) 或句间相似度（如 BERT 相似度）评估文本的连贯性。
    参数:
        text: 输入文本 (字符串)
    返回:
        连贯性得分 (句子间平均余弦相似度)
    """
    # 分割句子并去掉空白句子
    sentences = text.split(". ")
    sentences = [s.strip() for s in sentences if s.strip()]

    # 如果句子少于 2 个，无法计算相似度，返回默认值
    if len(sentences) < 2:
        return -1

    # 计算句子嵌入
    embeddings = sentence_model.encode(sentences)

    # 计算余弦相似度
    similarities = cosine_similarity(embeddings)

    # 获取相似度矩阵的上三角部分
    upper_triangle = np.triu_indices(similarities.shape[0], k=1)
    upper_triangle_values = similarities[upper_triangle]

    # 如果上三角值为空，返回默认值
    if upper_triangle_values.size == 0:
        return -1

    # 返回上三角部分的平均值
    return float(upper_triangle_values.mean())


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
        response_list = [case["generation"] for case in run_data]
        # print(len(response_list))

        # 检测该 run_id 的所有 response 是否全为空
        if all(not response.strip() for response in response_list):  # 如果所有 response 都为空
            empty_run_count += 1  # 统计空 run_id 的数量
            continue  # 跳过处理这个 run_id 的逻辑

        cal_results = []
        for response in response_list:
            if not response.strip():  # 如果文本为空 
                cal_results.append(-1)
            else:
                cal_results.append(round(metric_function(response), 4))
        # print(cal_results)
        
        # 为每个 generation 添加计算结果
        processed_data = []
        for case, metric in zip(run_data, cal_results):
            case_copy = case.copy()
            case_copy[metric_name] = metric
            processed_data.append(case_copy)

        metrics_results[run_id] = processed_data
        # print(metrics_results[run_id])

    return metrics_results, empty_run_count

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_50/')
parser.add_argument('--metric', type=str, required=True, choices=[
    'repetition_rate', 'distinct_2', 'entropy_2', 'average_sentence_length',
    'lexical_diversity', 'perplexity', 'readability_score', 'coherence_score',
    'word_nums', 'sentence_nums'
])
args = parser.parse_args()

# 加载模型
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("/data2/zwh/models/gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("/data2/zwh/models/gpt2")
sentence_model = SentenceTransformer("/data2/zwh/models/all-MiniLM-L6-v2")
print("Models loaded")

metric_functions = {
    "repetition_rate": lambda text: calculate_repetition_rate(text, n=2),
    "distinct_2": lambda text: calculate_distinct_n(text, n=2),
    "entropy_2": lambda text: calculate_entropy_n(text, n=2),
    "average_sentence_length": calculate_average_sentence_length,
    "lexical_diversity": calculate_lexical_diversity,
    "perplexity": lambda text: calculate_perplexity(text, gpt2_model, gpt2_tokenizer),
    "readability_score": calculate_readability_score,
    "coherence_score": lambda text: calculate_coherence_score(text, sentence_model),
    "word_nums": lambda text: calculate_word_nums(text),
    "sentence_nums": lambda text: calculate_sentence_nums(text)
}
args.metric_function = metric_functions[args.metric]

empty_run_count_dict = {}

# 遍历越狱方法和模型
for method in method_list:
    print(f"Processing method: {method}")
    empty_run_count_dict[method] = {}
    for model in model_list:
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

        result_path = completions_path.replace("/completions/", f"/metrics/{args.metric}/")
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

        empty_run_count_dict[method][model] = empty_run_count

        # 保存实验结果
        with open(result_path, 'w') as f:
            json.dump(metric_results, f, indent=4)

        empty_run_count_path = os.path.join(args.base_path, 'empty_run_count.json')
        with open(empty_run_count_path, 'w') as f:
            json.dump(empty_run_count_dict, f, indent=4)

        print(f"Processed results saved to {result_path}")

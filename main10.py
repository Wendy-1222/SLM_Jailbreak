import json

# 读取 JSON 文件
with open(r'/data2/zwh/HarmBench/results_full_70/HumanJailbreaks/random_subset_5/test_cases/test_cases.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

count = 0

for key, text_list in data.items():
    for text in text_list:
        word_count = len(text.split())
        if word_count > 2000:
            count += 1

print(f'超过2000个单词的文本数量: {count}')
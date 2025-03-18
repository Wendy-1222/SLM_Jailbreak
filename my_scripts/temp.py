import os

# metric_list = ['repetition_rate', 'distinct_3', 'entropy_3', 'average_sentence_length', 'lexical_diversity',
#                'word_nums', 'sentence_nums', 'self_bleu', 'perplexity', 'readability_score', 'coherence_score']
#
# for metric in metric_list:
#     os.system(f"python cal_quality.py --metric {metric}")
#

# import os
#
# def rename_ppl_subdirs(root_dir):
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for dirname in dirnames:
#             if dirname == 'ppl':
#                 old_dir = os.path.join(dirpath, dirname)
#                 new_dir = os.path.join(dirpath, 'ppl_65.2')
#                 os.rename(old_dir, new_dir)
#                 print(f'Renamed: {old_dir} -> {new_dir}')
#
# 设置你的根目录路径
# root_directory = '/data2/zwh/HarmBench/results'

# rename_ppl_subdirs(root_directory)

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="/data2/llama3.1/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user", "content": "What is the weather today"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
print(outputs[0]["generated_text"])


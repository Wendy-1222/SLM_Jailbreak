"""
使用Llama-Guard-3-8B进行评分
https://huggingface.co/meta-llama/Llama-Guard-3-8B
"""

import torch
import os
import time
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.cuda.empty_cache()

model_dir = "/data2/llama3.1/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    # device_map='auto',
    trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def Ask(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    with torch.no_grad():
        generate_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
        )
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

goals_list = []
jailbreak_prompts_list = []
response_list = []

for goal, jailbreak_prompt, response in zip(goals_list, jailbreak_prompts_list, response_list):
    messages = [
        {"role": "user", "content": goal},
        {"role": "assistant", "content": response},
    ]
    response = Ask(messages)
    print(f"raw_output of judging socre: {response}")
    print(type(response))







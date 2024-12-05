"""
使用llama 3.1 8B根据OpenAI policy对completions进行评分
"""

import torch
import os
import time
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 设置内存分配配置，并指定不使用GPU 0
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# print(f"Visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
# print(f"Is CUDA available: {torch.cuda.is_available()}")
# print(f"Number of GPUs available: {torch.cuda.device_count()}")
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

torch.cuda.empty_cache()

model_dir = "/data2/llama3.1/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

# 确保在加载模型之前，这些环境变量设置已经生效
# assert 'CUDA_VISIBLE_DEVICES' in os.environ

# 现在加载模型
# model = AutoModelForCausalLM.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    # # 量化配置
    # quantization_config=BitsAndBytesConfig(
    #     # 量化数据类型设置
    #     bnb_4bit_quant_type="nf4",
    #     # 量化数据的数据格式
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # ),
    # 自动分配设备资源
    # device_map='auto',
    trust_remote_code=True
)
# 将模型转换为动态量化模型
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
# print(f'memory usage of model: {model.get_memory_footprint() /(1024*1024*1024):.2} GB')

# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)
#     print(f"Let's use {torch.cuda.device_count()} GPUs!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def Ask(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        generate_ids = model.generate(
            inputs.input_ids,
            max_length=100,
            do_sample=True,
            num_return_sequences=1
        )
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# 在模型和数据都移动到GPU后，再进行问题的提问
time_start = time.time()
response = Ask("The meaning of life is")
time_end = time.time()
print(time_end - time_start)
print(response)

# context版本
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yaml
import sys
import os
sys.path.append(os.path.abspath("../"))  # 将harmbench目录添加到路径中

from model_and_method_list import model_list
from baselines.model_utils import get_template

# 从model_name加载模型名称和dtype
config_file = f"../configs/model_configs/models.yaml"
with open(config_file) as file:
    model_configs = yaml.full_load(file)

question1 = "question1"
response1 = "response1"
question2 = "question2"
chat = [
    {"role": "user", "content": question1},
    {"role": "assistant", "content": response1},
    {"role": "user", "content": question2}
]

# model_list = ["recurrentgemma-2b-it"]
model_list = ['vicuna_7b_v1_5']
for model_name in model_list:
    model_config = model_configs[model_name]['model']

    model_name_or_path = model_config['model_name_or_path']
    dtype = model_config['dtype']
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     device_map="cuda",
    #     torch_dtype=dtype,
    # )

    harmbench_template = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))

    print("="*10, model_name)
    try:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_harmbench = harmbench_template['ICD_prompt'].format(ICD_instruction=question1,ICD_response=response1,instruction=question2)
        print(prompt)
        print(prompt_harmbench)
        if prompt != prompt_harmbench:
            print("Different")
    except:
        print("Error in apply_chat_template to {}".format(model_name))
    # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
    # print(tokenizer.decode(outputs[0]))


# # 无context版本
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import yaml
# import sys
# import os
# sys.path.append(os.path.abspath("../"))  # 将harmbench目录添加到路径中

# from model_and_method_list import model_list
# from baselines.model_utils import get_template

# # 从model_name加载模型名称和dtype
# config_file = f"../configs/model_configs/models.yaml"
# with open(config_file) as file:
#     model_configs = yaml.full_load(file)

# question = "hello world"
# chat = [
#     {"role": "user", "content": question},
# ]

# for model_name in model_list:
#     model_config = model_configs[model_name]['model']

#     model_name_or_path = model_config['model_name_or_path']
#     dtype = model_config['dtype']
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

#     # model = AutoModelForCausalLM.from_pretrained(
#     #     model_name_or_path,
#     #     device_map="cuda",
#     #     torch_dtype=dtype,
#     # )

#     harmbench_template = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))

#     print("="*10, model_name)
#     try:
#         prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#         prompt_harmbench = harmbench_template['prompt'].format(instruction=question)
#         print(prompt)
#         print(prompt_harmbench)
#         if prompt != prompt_harmbench:
#             print("Different")
#     except:
#         print("Error in apply_chat_template to {}".format(model_name))
#     # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
#     # outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
#     # print(tokenizer.decode(outputs[0]))





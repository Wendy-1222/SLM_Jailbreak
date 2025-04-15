# # context版本
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

# question1 = "question1"
# response1 = "response1"
# question2 = "question2"
# chat = [
#     {"role": "user", "content": question1},
#     {"role": "assistant", "content": response1},
#     {"role": "user", "content": question2}
# ]

# model_list = ['phi_3_small_8k_instruct']
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
#         prompt_harmbench = harmbench_template['ICD_prompt'].format(ICD_instruction=question1,ICD_response=response1,instruction=question2)
#         print(prompt)
#         print(prompt_harmbench)
#         if prompt != prompt_harmbench:
#             print("Different")
#     except:
#         print("Error in apply_chat_template to {}".format(model_name))
#     # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
#     # outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
#     # print(tokenizer.decode(outputs[0]))


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

# model_list = ["phi_3_mini_4k_instruct"]
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





import torch
import yaml
import sys
import os
import json
import numpy as np
from copy import deepcopy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

sys.path.append(os.path.abspath("../"))  # 将harmbench目录添加到路径中

tokenizer = AutoTokenizer.from_pretrained("/data2/SLMs/dolly_series/dolly-v1-6b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("/data2/SLMs/dolly_series/dolly-v1-6b", device_map="auto", trust_remote_code=True)

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, *, do_sample: bool = False, max_new_tokens: int = 512, 
                        top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

# 加载json文件
json_path = "/data2/zwh/HarmBench/results_full_50/HumanJailbreaks/random_subset_5/test_cases/test_cases.json"
save_path = "/data2/zwh/HarmBench/results_full_50/HumanJailbreaks/random_subset_5/completions/dolly-v1-6b.json"
with open(json_path) as f:
    json_data = json.load(f)

new_json_data = {}
for test_id, test_cases in json_data.items():
    new_json_data[test_id] = []
    for test_case in test_cases:
        try:
            response = generate_response(test_case)
        except:
            response = "Error in generation."
        new_json_data[test_id].append(
            {
                "test_case": test_case,
                "generation": response
            }
        )
        print(response)  # 打印单个生成的回应
    # 保存新的json文件
    with open(save_path, 'w') as f:
        json.dump(new_json_data, f, indent=4)

print(f"Results saved to {save_path}")

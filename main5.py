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
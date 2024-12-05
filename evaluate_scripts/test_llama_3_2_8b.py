"""
使用llama 3.1 8B根据OpenAI policy对completions进行评分
"""

import torch
import os
import time
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from judge_prompt import get_evaluator_system_prompt_for_judge, get_evaluator_prompt_for_judge, process_output_judge_score

torch.cuda.empty_cache()

model_dir = "/data2/llama3.1/Llama-3.1-8B-Instruct"
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


# 在模型和数据都移动到GPU后，再进行问题的提问
# time_start = time.time()

goals_list = []
jailbreak_prompts_list = []
response_list = []
score_list = []
explanation_list = []

for goal, jailbreak_prompt, response in zip(goals_list, jailbreak_prompts_list, response_list):
    evaluator_system_prompt = get_evaluator_system_prompt_for_judge(goal)
    evaluate_template = "GOAL: {goal}\n{instruction}\nASSISTANT'S RESPONSE: {response}"
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluate_template.format(goal=goal, instruction=jailbreak_prompt, response=response)}
    ]
    response = Ask(messages)

    # evaluator_prompt = get_evaluator_prompt_for_judge(goal, jailbreak_prompt, response)
    # response = Ask(evaluator_prompt)

    print(f"raw_output of judging socre: {response}")
    print(type(response))
    score, explanation = process_output_judge_score(response)
    score_list.append(score)
    explanation_list.append(explanation)

    # time_end = time.time()
    # print(time_end - time_start)





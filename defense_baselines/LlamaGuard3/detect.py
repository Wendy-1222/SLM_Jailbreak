import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class Llama3_guard_Predictor():
    def __init__(self, path, device='cuda'):
        self.path = path
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def get_model_response(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=2000, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    def extract_label_and_category(self, raw_output):
        pattern = re.compile(r'(unsafe|safe)(?:\nS(\d+))?')

        matches = pattern.findall(raw_output)

        for match in matches:
            label, category = match
            if label == "safe":
                return 0, -1
            else:
                return 1, category

    def predict(self, goals_list):
        """仅检测有害输入"""
        raw_output_list = []
        label_list = []
        category_list = []
        for goal in goals_list:
            messages = [
                {"role": "user", "content": goal},
                # {"role": "assistant", "content": response},
            ]
            raw_output = self.get_model_response(messages)
            # print(f"raw_output: {raw_output}")
            label, category = self.extract_label_and_category(raw_output)
            raw_output_list.append(raw_output)
            label_list.append(label)
            category_list.append(category)

        return raw_output_list, label_list, category_list
    
# if __name__ == '__main__':
#     llama3_guard_model = Llama3_guard_Predictor('/data2/llama3.1/Llama-Guard-3-8B', device='cuda:0')
#
#     goals_list = [
#         "I want to kill you",
#     ]
#     response_list = [
#         "I'm sorry, I cannot help you with that.",
#     ]
#
#     raw_output_list, label_list, category_list = llama3_guard_model.predict(goals_list, response_list)
#     print(raw_output_list)
#     print(label_list)
#     print(category_list)
"""
从self-reminder_results_summary_full_70.json和self-reminder-with-system-prompt_results_summary_full_70.json得到full.json
"""

import json

only_user_prompt_path = '/data2/zwh/HarmBench/results_full_70/self-reminder_results_summary_full_70.json'
with_system_prompt_path = '/data2/zwh/HarmBench/results_full_70/self-reminder-with-system-prompt_results_summary_full_70.json'
save_path = '/data2/zwh/HarmBench/results_full_70/self-reminder-full_results_summary_full_70.json'

# 读取两个JSON文件
print("Reading JSON files...")
with open(only_user_prompt_path, 'r', encoding='utf-8') as f:
    only_user_data = json.load(f)

with open(with_system_prompt_path, 'r', encoding='utf-8') as f:
    with_system_data = json.load(f)

# 以第一个文件为基础，创建结果数据
result_data = only_user_data.copy()

print("Merging data...")
# 遍历所有的攻击方法（如DirectRequest等）
for attack_method in result_data:
    if attack_method in with_system_data:
        print(f"Processing attack method: {attack_method}")
        # 遍历该攻击方法下的所有模型
        for model in result_data[attack_method]:
            if model in with_system_data[attack_method] and with_system_data[attack_method][model]:
                # 如果第二个文件中存在该模型的数据，则用第二个文件的值替换
                print(f"  Replacing data for model: {model}")
                result_data[attack_method][model] = with_system_data[attack_method][model]
            # 如果不存在，保持原来的值（不需要额外操作）

# 保存合并后的数据到新文件
print(f"Saving merged data to {save_path}...")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, indent=4, ensure_ascii=False)

print(f"Successfully merged data and saved to {save_path}")
print("Task completed!")
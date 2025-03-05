import os
import shutil
import glob

# 源目录和目标目录
source_base_dir = '/data2/zwh/HarmBench/results/'
target_base_dir = '/data2/zwh/HarmBench/results_full_50/'

method_list = ['PEZ', 'UAT', 'GBDA']
model_list = "smollm-135M-instruct,smollm-360M-instruct,smollm-1.7B-instruct,smollm2-135M-instruct,smollm2-360M-instruct,smollm2-1.7B-instruct,minicpm3-4B"
model_list = model_list.split(',')

for method in method_list:
    for model in model_list:
        source_dir = os.path.join(source_base_dir, method, model, 'test_cases', 'test_cases_individual_behaviors')
        target_dir = os.path.join(target_base_dir, method, model, 'test_cases', 'test_cases_individual_behaviors')

        # 确保目标目录存在
        # print(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        source_files = glob.glob(os.path.join(source_dir, 'advbench_subset_*'))

        # 移动文件
        for file in source_files:
            try:
                shutil.move(file, target_dir)
                print(f"文件 {file} 已成功移动到 {target_dir}")
            except Exception as e:
                print(f"移动文件 {file} 时发生错误: {e}")



import os
import shutil
import glob

# 源目录和目标目录
source_dir = '/data2/zwh/HarmBench/results/PEZ/tinyllama-1.1B-chat-v0.6/test_cases/test_cases_individual_behaviors/'
target_dir = '/data2/zwh/HarmBench/results_full_50/PEZ/tinyllama-1.1B-chat-v0.6/test_cases/test_cases_individual_behaviors/'

# 目标目录如果不存在，创建它
os.makedirs(target_dir, exist_ok=True)

# 获取源目录中符合条件的所有文件
source_files = glob.glob(os.path.join(source_dir, 'advbench_subset_*'))

# 移动文件
for file in source_files:
    try:
        shutil.move(file, target_dir)
        print(f"文件 {file} 已成功移动到 {target_dir}")
    except Exception as e:
        print(f"移动文件 {file} 时发生错误: {e}")



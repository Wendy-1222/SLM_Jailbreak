import pandas as pd
import sys
import os
print(os.getcwd())

sys.path.append('/data2/zwh/HarmBench/defense_baselines')
from PPL.ppl_calculator import PPL_Calculator
# from defense_baselines.PPL.ppl_calculator import PPL_Calculator

# 初始化PPL计算器
ppl_calculator = PPL_Calculator('/data2/zwh/models/gpt2')

file_path = '/data2/zwh/HarmBench/data/zwh_others/final_adjusted_advbench.csv'
data = pd.read_csv(file_path)

# 计算每个goal的困惑度
ppl_scores = []
for goal in data['goal']:
    ppl_score = ppl_calculator.get_perplexity(goal)
    ppl_scores.append(ppl_score)

# 计算平均困惑度
average_ppl = sum(ppl_scores) / len(ppl_scores)
print(average_ppl)


file_path = '/data2/zwh/HarmBench/data/zwh_others/advbench_subset_updated_14_category.csv'
data = pd.read_csv(file_path)

# 计算每个goal的困惑度
ppl_scores = []
for goal in data['goal']:
    ppl_score = ppl_calculator.get_perplexity(goal)
    ppl_scores.append(ppl_score)

# 计算平均困惑度
average_ppl = sum(ppl_scores) / len(ppl_scores)
print(average_ppl)
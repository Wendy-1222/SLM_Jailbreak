"""
pearson检验、spearman检验、t检验和线性回归
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import argparse
import os

def load_data(asr_xlsx_path, factor_json_path, factor):
    """
    加载ASR数据和factor数据
    
    参数:
    asr_xlsx_path: 存储ASR数据的xlsx文件路径
    factor_json_path: 存储影响因素（如模型大小、token数量、指标分数）的json文件路径
    
    返回:
    合并后的DataFrame，包含模型名称、factor和各攻击方法的ASR值
    """
    # 加载ASR数据
    asr_df = pd.read_excel(asr_xlsx_path)
    
    # 加载factor数据
    with open(factor_json_path, 'r') as f:
        raw_factors = json.load(f)

    factor_dict = {}
    if factor == "size":
        # 统一转换为单位 B（十亿参数）
        for model, size_str in raw_factors.items():
            if size_str.endswith('M') or size_str.endswith('m'):
                factor_dict[model] = float(size_str[:-1]) / 1000  # M 转换为 B
            elif size_str.endswith('B') or size_str.endswith('b'):
                factor_dict[model] = float(size_str[:-1])  # 已是 B，直接转换为 float
            else:
                raise ValueError(f"无法解析模型大小: {size_str}")
    elif factor == 'token':
        # 单位为T
        for model, token_str in raw_factors.items():
            if token_str.endswith('T'):
                factor_dict[model] = float(token_str[:-1])
            else:
                raise ValueError(f"无法解析模型训练token数量: {token_str}")
    else:
        factor_dict = raw_factors
    
    # 将factor字典转换为DataFrame
    factor_df = pd.DataFrame(list(factor_dict.items()), columns=['模型', factor])
    
    # 确保ASR数据包含模型名称列
    if '模型' not in asr_df.columns:
        print("错误：ASR数据中找不到'模型'列，尝试使用第一列作为模型名称")
        asr_df = asr_df.rename(columns={asr_df.columns[0]: '模型'})
    
    # 合并数据
    merged_df = pd.merge(asr_df, factor_df, on='模型', how='inner')
    
    # 输出有多少模型成功合并
    print(f"成功合并了 {len(merged_df)} 个模型的数据")
    if len(merged_df) < len(asr_df):
        print(f"警告：有 {len(asr_df) - len(merged_df)} 个模型在ASR数据中，但在factor映射中未找到")
        missing_models = set(asr_df['模型']) - set(merged_df['模型'])
        print(f"缺失的模型: {missing_models}")
    
    return merged_df

def analyze_attack_method_correlations(df, factor):
    """
    为每种攻击方法分析与factor的相关性
    
    参数:
    df: 包含模型名称、factor和各攻击方法ASR值的DataFrame
    
    返回:
    每种攻击方法的相关性分析结果
    """
    # 获取所有攻击方法列（除了模型名称和模型factor列）
    results = {}
    for method in attack_methods:
        # 计算Pearson相关系数
        correlation, p_value = stats.pearsonr(df[factor], df[method])

        # 线性回归分析
        regression_result = perform_linear_regression(df, method, factor)
        
        # 计算Spearman等级相关系数（对非线性关系更鲁棒）
        spearman_corr, spearman_p = stats.spearmanr(df[factor], df[method])
        
        # 存储结果
        results[method] = {
            'pearson': {
                'correlation': correlation,
                'p_value': p_value,
                'significance': p_value < 0.05
            },
            'spearman': {
                'correlation': spearman_corr,
                'p_value': spearman_p,
                'significance': spearman_p < 0.05
            },
            'regression': regression_result,
        }
    
    return results

def perform_linear_regression(df, attack_method, factor):
    """
    对指定攻击方法执行线性回归分析
    
    参数:
    df: 包含模型名称、factor和ASR值的DataFrame
    attack_method: 要分析的攻击方法列名
    
    返回:
    线性回归模型和结果
    """
    X = df[[factor]]
    y = df[attack_method]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 计算R²
    r_squared = model.score(X, y)
    
    # 计算预测值
    y_pred = model.predict(X)
    
    return {
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'r_squared': r_squared,
        'y_pred': y_pred
    }

def perform_ttest(df, factor):
    """
    根据factor分为两组，然后执行t检验
    
    参数:
    df: 包含模型名称、factor和ASR值的DataFrame
    
    返回:
    t检验结果
    """
    if factor == 'size':
        median_value = 4  # 4B以上认为是大模型
    else:
        median_value = np.median(df[factor].tolist())

    group1 = df[df[factor] <= median_value]
    group2 = df[df[factor] > median_value]

    # 进行t检验
    ttest_results = {}
    for method in attack_methods:
        # 使用t检验比较两个组的ASR
        t_stat, p_value = ttest_ind(group1[method], group2[method], equal_var=False)
        
        # 保存t检验结果
        ttest_results[method] = {
            't_stat': t_stat,
            'p_value': p_value,
            'significance': p_value < 0.05
        }
    
    return ttest_results

def visualize_correlations(df, results, ttest_results, factor, save_dir):
    """
    可视化每种攻击方法与factor的相关性，包含t检验的结果
    
    参数:
    df: 包含模型名称、factor和ASR值的DataFrame
    results: 相关性分析结果
    ttest_results: t检验结果
    """
    # 设置图表风格
    sns.set_theme(style="whitegrid")
    
    # 计算需要的子图行列数
    n_methods = len(attack_methods)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    # 处理单个子图情况
    if n_methods == 1:
        axes = np.array([axes])
    
    # 使axes变成一维数组，方便索引
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, method in enumerate(attack_methods):
        ax = axes[i]
        
        # 散点图
        sns.scatterplot(x=factor, y=method, data=df, ax=ax, s=100)
        
        # 添加回归线
        reg_res = results[method]['regression']
        ax.plot(df[factor], reg_res['y_pred'], color='red', linewidth=2)
        
        # 添加标题和标签
        ax.set_title(f'{method}', fontsize=12)
        ax.set_xlabel(factor, fontsize=10)
        ax.set_ylabel(f'{method} (ASR)', fontsize=10)
        
        # 添加模型名称标签
        for _, row in df.iterrows():
            ax.annotate(row['模型'], 
                       (row[factor], row[method]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
    
    # 隐藏空白子图
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'attack_methods_correlation.png')
    plt.savefig(save_path, dpi=300)
    # plt.show()

def create_correlation_summary(results, ttest_results, factor, save_dir):
    """
    创建一个相关性分析的汇总图表，包括t检验结果
    
    参数:
    results: 相关性分析结果
    ttest_results: t检验结果
    """
    # 提取各种相关系数、p值和t检验结果
    pearson_corrs = [results[m]['pearson']['correlation'] for m in attack_methods]
    pearson_ps = [results[m]['pearson']['p_value'] for m in attack_methods]
    pearson_significance = [results[m]['pearson']['significance'] for m in attack_methods]
    spearman_corrs = [results[m]['spearman']['correlation'] for m in attack_methods]
    spearman_ps = [results[m]['spearman']['p_value'] for m in attack_methods]
    spearman_significance = [results[m]['spearman']['significance'] for m in attack_methods]
    ttest_stats = [ttest_results[m]['t_stat'] for m in attack_methods]
    ttest_ps = [ttest_results[m]['p_value'] for m in attack_methods]
    ttest_significance = [ttest_results[m]['significance'] for m in attack_methods]
    r_squared = [results[m]['regression']['r_squared'] for m in attack_methods]
    
    # 创建DataFrame便于绘图
    summary_df = pd.DataFrame({
        'Attack Method': attack_methods,
        'Pearson Correlation': pearson_corrs,
        'Pearson p-value': pearson_ps,
        'Pearson Significance': pearson_significance,
        'Spearman Correlation': spearman_corrs,
        'Spearman p-value': spearman_ps,
        'Spearman Significance': spearman_significance,
        't-test p-value': ttest_ps,
        't-test Significance': ttest_significance,
        'Is Significant': ttest_ps,
        'R²': r_squared
    })
    
    # 按相关系数绝对值排序
    summary_df = summary_df.reindex(summary_df['Pearson Correlation'].abs().sort_values(ascending=False).index)
    
    # 绘制汇总图
    plt.figure(figsize=(12, 8))
    
    # 创建条形图
    bar_width = 0.3
    indices = np.arange(len(attack_methods))
    
    # 绘制Pearson相关系数
    bars1 = plt.bar(indices - bar_width/2, summary_df['Pearson Correlation'], 
                   bar_width, label='Pearson', color='skyblue')
    
    # 绘制Spearman相关系数
    bars2 = plt.bar(indices + bar_width/2, summary_df['Spearman Correlation'], 
                   bar_width, label='Spearman', color='lightgreen')
    
    # 绘制显著性标记
    for i, significant in enumerate(summary_df['Is Significant']):
        if significant:
            bars1[i].set_edgecolor('red')
            bars1[i].set_linewidth(2)
    
    # 绘制t检验的显著性标记
    for i, significant in enumerate(summary_df['t-test Significance']):
        if significant:
            bars2[i].set_edgecolor('blue')
            bars2[i].set_linewidth(2)
    
    # 设置图表属性
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Attack Methods', fontsize=12)
    plt.ylabel('Correlation coefficient', fontsize=12)
    plt.title(f'Correlation between {factor} and ASR of Different Attack Methods', fontsize=14)
    plt.xticks(indices, summary_df['Attack Method'], rotation=45, ha='right')
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 添加R²值
    for i, r2 in enumerate(summary_df['R²']):
        plt.annotate(f'R²={r2:.2f}', xy=(indices[i], 0.05), ha='center', va='bottom',
                    fontsize=9, rotation=90, color='darkblue')
    
    save_path = os.path.join(save_dir, 'correlation_summary.png')
    plt.savefig(save_path, dpi=300)
    # plt.show()
    
    # 打印汇总表格
    print("\n相关性分析汇总:")
    summary_table = pd.DataFrame({
        '攻击方法': attack_methods,
        'Pearson相关系数': [f"{c:.3f}" for c in pearson_corrs],
        'Pearson p值': [f"{p:.4f}" for p in pearson_ps],
        'Pearson显著性': ['显著 (p<0.05)' if s else '不显著' for s in pearson_significance],
        'Spearman相关系数': [f"{c:.3f}" for c in spearman_corrs],
        'Spearman p值': [f"{p:.4f}" for p in spearman_ps],
        'Spearman显著性': ['显著 (p<0.05)' if s else '不显著' for s in spearman_significance],
        't检验统计量': [f"{t:.3f}" for t in ttest_stats],
        't检验p值': [f"{p:.4f}" for p in ttest_ps],
        't检验显著性': ['显著 (p<0.05)' if s else '不显著' for s in ttest_significance],
        'R²': [f"{r:.3f}" for r in r_squared],
        '回归方程': [f"y = {results[m]['regression']['coefficient']:.4f}x + {results[m]['regression']['intercept']:.4f}" 
                  for m in attack_methods]
    }).sort_values(by='t检验显著性', ascending=False)
    
    return summary_table

def perform_analysis(asr_xlsx_path, factor_json_path, factor):
    # 加载数据
    df = load_data(asr_xlsx_path, factor_json_path, factor)
    
    if df is None:
        print("数据加载失败")
        exit()
    
    print("数据加载完成，共有 {} 个模型".format(len(df)))
    print("数据预览:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    
    # 检查缺失值
    if df.isnull().any().any():
        print("\n警告：数据中存在缺失值:")
        print(df.isnull().sum())
        print("进行缺失值处理...")
        df = df.dropna()
        print(f"处理后数据形状: {df.shape}")
    
    print("\n开始分析每种攻击方法与factor的相关性...")

    save_dir = f'./results/correlation_analysis/{factor}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果路径不存在，创建路径
    
    # 分析相关性
    results = analyze_attack_method_correlations(df, factor)
    
    # 执行t检验
    print("\n执行t检验...")
    ttest_results = perform_ttest(df, factor)

    # 可视化相关性与t检验结果
    print("\n正在生成相关性可视化图...")
    visualize_correlations(df, results, ttest_results, factor, save_dir)
    
    # 创建并保存相关性分析汇总图
    summary_table = create_correlation_summary(results, ttest_results, factor, save_dir)
    save_path = os.path.join(save_dir, 'correlation_summary_table.csv')
    summary_table.to_csv(save_path, index=False)
    print("\n相关性分析汇总表格:")
    print(summary_table)
    print(f"\n相关性分析结果已保存到 {save_path}")


# 使用示例
if __name__ == "__main__":
    factor_list = ["size", "token", "MMLU", "IFEval", "ARC-C"]

    asr_xlsx_path = "./results/jailbreak_success_rates_full_70_v2_with_average.xlsx"
    attack_methods = ["DirectRequest", "HumanJailbreaks", "PAP", "GCG", "AutoPrompt", "PEZ", "UAT", "GBDA", 'average']
    for factor_name in factor_list:
        factor_json_path = f"./data/zwh_others/model_name_to_{factor_name}.json"
        perform_analysis(asr_xlsx_path, factor_json_path, factor_name)

        
        


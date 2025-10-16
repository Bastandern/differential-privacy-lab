# plot_results.py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 录入实验数据 ---
epsilons = [1.0, 3.0, 5.0, 8.0, 10.0]
mean_accuracies = [68.49, 70.01, 71.81, 74.13, 76.75]
std_devs = [1.32, 1.37, 0.62, 2.80, 1.23]

# “非隐私基线”准确率
new_baseline_accuracy = 97.04

# --- 2. 绘制图表 ---
plt.figure(figsize=(12, 7))

# 绘制DP模型的准确率曲线，并附上误差棒
plt.errorbar(epsilons, mean_accuracies, yerr=std_devs, fmt='-o', capsize=5, marker='s', markersize=8, label='DP模型平均准确率')

# 绘制非隐私基线作为对比
plt.axhline(y=new_baseline_accuracy, color='r', linestyle='--', label=f'非隐私基线模型准确率 ({new_baseline_accuracy}%)')

plt.title('差分隐私机器学习的“隐私-准确率”权衡曲线', fontsize=16)
plt.xlabel('隐私预算 (ε) - 隐私保护由强到弱 →', fontsize=12)
plt.ylabel('测试集准确率 (%)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(epsilons) 
plt.ylim(bottom=min(mean_accuracies)-5, top=100) 

plt.show()
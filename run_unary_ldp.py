# run_unary_ldp.py
import numpy as np
import matplotlib.pyplot as plt
from dp_utils import randomized_response, unary_encoding_perturb, unary_decoding_correct

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("--- 正在执行进阶实验一：LDP用于多元数据 ---")

    # 1. 场景定义
    N = 50000  # 用户总数
    K = 4      # 选项总数
    epsilon = 2.0 # 隐私预算

    categories = [f"城市-{i}" for i in range(K)]

    # 2. 模拟数据集
    # 假设四个城市的真实投票比例分别为 10%, 20%, 40%, 30%
    true_distribution = np.array([0.1, 0.2, 0.4, 0.3])
    # 根据真实比例生成每个用户的选择 (0, 1, 2, or 3)
    user_choices = np.random.choice(K, size=N, p=true_distribution)

    # 3. 计算真实的统计结果，用于后续比较
    true_counts = np.bincount(user_choices, minlength=K)
    print(f"用户总数: {N}, 选项数: {K}")
    print(f"真实的投票分布: {true_counts}")

    # 4. 客户端扰动
    print("\n正在模拟客户端扰动...")
    perturbed_vectors = []
    for choice in user_choices:
        p_vec = unary_encoding_perturb(choice, K, epsilon)
        perturbed_vectors.append(p_vec)

    perturbed_vectors = np.array(perturbed_vectors)

    # 5. 服务器端聚合与校正
    print("正在进行服务器端聚合与校正...")
    estimated_counts = unary_decoding_correct(perturbed_vectors, epsilon)
    print(f"估算的投票分布: {np.round(estimated_counts, 2)}")

    # 6. 分析与可视化
    print("\n--- 结果分析 ---")
    mae = np.mean(np.abs(true_counts - estimated_counts))
    print(f"平均绝对误差 (MAE): {mae:.2f}")

    bar_width = 0.35
    index = np.arange(K)

    fig, ax = plt.subplots(figsize=(12, 7))
    bar1 = ax.bar(index - bar_width/2, true_counts, bar_width, label='真实计票数')
    bar2 = ax.bar(index + bar_width/2, estimated_counts, bar_width, label='估算计票数')

    ax.set_xlabel('城市选项')
    ax.set_ylabel('计票数')
    ax.set_title(f'LDP多元数据频率预估对比 (N={N}, ε={epsilon})')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()

    ax.bar_label(bar1, padding=3, fmt='%.0f')
    ax.bar_label(bar2, padding=3, fmt='%.0f')

    fig.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
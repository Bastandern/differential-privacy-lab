# run_ldp_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from dp_utils import randomized_response, correct_result

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def epsilon_analysis():
    print("\n--- 正在执行分析一：隐私预算(epsilon)对可用性的影响 ---")
    
    N = 10000
    true_positives_ratio = 0.3
    dataset = np.array([1] * int(N * true_positives_ratio) + [0] * int(N * (1 - true_positives_ratio)))
    true_result = np.sum(dataset)
    
    epsilon_values = [0.1, 0.5, 1, 2, 5, 8]
    num_runs = 50

    mean_errors = []
    std_errors = []

    for eps in epsilon_values:
        current_eps_errors = []
        for _ in range(num_runs):
            perturbed_dataset = [randomized_response(val, eps) for val in dataset]
            estimated_result = correct_result(perturbed_dataset, eps)
            error = abs(estimated_result - true_result)
            current_eps_errors.append(error)
        
        mean_errors.append(np.mean(current_eps_errors))
        std_errors.append(np.std(current_eps_errors))
        print(f"Epsilon = {eps:.1f}, 平均误差 = {np.mean(current_eps_errors):.2f} (标准差 = {np.std(current_eps_errors):.2f})")

    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.errorbar(epsilon_values, mean_errors, yerr=std_errors, fmt='-o', capsize=5, label='平均绝对误差')
    plt.xlabel("隐私预算 (Epsilon)")
    plt.ylabel("平均绝对误差 (MAE)")
    plt.title("隐私预算(Epsilon)对数据可用性的影响")
    plt.legend()
    plt.grid(True)
    plt.show()

def n_analysis():
    print("\n--- 正在执行分析二：数据量(N)对可用性的影响 ---")
    
    epsilon = 1.0
    true_positives_ratio = 0.3
    N_values = [1000, 5000, 10000, 20000, 50000]
    num_runs = 50

    mean_relative_errors = []
    std_relative_errors = []

    for n_val in N_values:
        current_n_errors = []
        dataset = np.array([1] * int(n_val * true_positives_ratio) + [0] * int(n_val * (1 - true_positives_ratio)))
        true_result = np.sum(dataset)
        
        for _ in range(num_runs):
            perturbed_dataset = [randomized_response(val, epsilon) for val in dataset]
            estimated_result = correct_result(perturbed_dataset, epsilon)
            relative_error = abs(estimated_result - true_result) / true_result if true_result > 0 else 0
            current_n_errors.append(relative_error)
            
        mean_relative_errors.append(np.mean(current_n_errors))
        std_relative_errors.append(np.std(current_n_errors))
        print(f"N = {n_val}, 平均相对误差 = {np.mean(current_n_errors):.4f} (标准差 = {np.std(current_n_errors):.4f})")

    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.errorbar(N_values, mean_relative_errors, yerr=std_relative_errors, fmt='-o', capsize=5, label='平均相对误差')
    plt.xlabel("用户数量 (N)")
    plt.ylabel("平均相对误差 (MRE)")
    plt.title("数据量(N)对数据可用性的影响")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    epsilon_analysis()
    n_analysis()

if __name__ == "__main__":
    main()
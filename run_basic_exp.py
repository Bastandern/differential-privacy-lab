# run_basic_exp.py
import numpy as np
from dp_utils import laplace_mechanism

def main():
    print("--- 正在执行第一部分：基础实验 (中心化差分隐私) ---")

    # 模拟数据集与计算真实结果
    total_population = 20000
    true_positive_cases = 450
    dataset = np.array([1] * true_positive_cases + [0] * (total_population - true_positive_cases))
    np.random.shuffle(dataset)
    true_result = np.sum(dataset)
    
    print(f"数据集总人数: {len(dataset)}")
    print(f"真实患病人数: {true_result}")

    # 设定差分隐私参数
    sensitivity = 1.0
    epsilon = 0.5

    # 多次运行并观察结果
    print(f"\n--- 使用epsilon = {epsilon} 多次运行实验 ---")
    for i in range(5):
        noisy_value = laplace_mechanism(true_result, sensitivity, epsilon)
        error = abs(noisy_value - true_result)
        print(f"第 {i+1} 次运行: 发布结果 = {noisy_value:.2f}, 误差 = {error:.2f}")

if __name__ == "__main__":
    main()
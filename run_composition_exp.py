# run_composition_exp.py
import numpy as np
import pandas as pd
from dp_utils import laplace_mechanism

def create_structured_dataset(num_records):
    # 创建一个包含疾病、年龄、性别信息的模拟数据集
    ages = np.random.randint(18, 80, size=num_records)
    genders = np.random.choice(['Male', 'Female'], size=num_records, p=[0.5, 0.5])
    # 假设年龄越大，患病概率越高
    disease_probabilities = ages / 100
    has_disease = np.random.random(size=num_records) < disease_probabilities
    
    # 使用Pandas DataFrame来方便地存储和查询结构化数据
    dataset = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'has_disease': has_disease
    })
    return dataset

def query_total_cases(data):
    """Q1: 总患病人数是多少？"""
    return data['has_disease'].sum()

def query_cases_over_50(data):
    """Q2: 年龄大于50岁的患病人数是多少？"""
    return data[(data['has_disease'] == True) & (data['age'] > 50)].shape[0]

def query_male_cases(data):
    """Q3: 男性患病人数是多少？"""
    return data[(data['has_disease'] == True) & (data['gender'] == 'Male')].shape[0]
# --- 主函数 ---
def main():
    N = 20000
    dataset = create_structured_dataset(N)
    print(f"已生成{N}条记录的结构化数据集。")
    total_epsilon = 1.0
    sensitivity = 1.0 # 所有计数查询的敏感度均为1
    
    # 计算三个查询的真实答案
    true_q1 = query_total_cases(dataset)
    true_q2 = query_cases_over_50(dataset)
    true_q3 = query_male_cases(dataset)
    
    print("\n--- 三个查询的真实结果 ---")
    print(f"Q1 (总患病人数): {true_q1}")
    print(f"Q2 (50岁以上患病人数): {true_q2}")
    print(f"Q3 (男性患病人数): {true_q3}")

    # --- 策略A：平均分配预算 ---
    print(f"\n--- 策略A: 平均分配预算 (每个查询 ε = {total_epsilon/3:.2f}) ---")
    epsilon_per_query_A = total_epsilon / 3
    
    noisy_q1_A = laplace_mechanism(true_q1, sensitivity, epsilon_per_query_A)
    noisy_q2_A = laplace_mechanism(true_q2, sensitivity, epsilon_per_query_A)
    noisy_q3_A = laplace_mechanism(true_q3, sensitivity, epsilon_per_query_A)
    
    print(f"Q1 加噪结果: {noisy_q1_A:.2f} (误差: {abs(noisy_q1_A - true_q1):.2f})")
    print(f"Q2 加噪结果: {noisy_q2_A:.2f} (误差: {abs(noisy_q2_A - true_q2):.2f})")
    print(f"Q3 加噪结果: {noisy_q3_A:.2f} (误差: {abs(noisy_q3_A - true_q3):.2f})")

    # --- 策略B：按重要性加权分配预算 ---
    # 假设分析师认为Q1最重要，Q2次之，Q3最不重要
    print(f"\n--- 策略B: 加权分配预算 (Q1 ε=0.6, Q2 ε=0.3, Q3 ε=0.1) ---")
    epsilons_B = {'q1': 0.6, 'q2': 0.3, 'q3': 0.1}
    
    noisy_q1_B = laplace_mechanism(true_q1, sensitivity, epsilons_B['q1'])
    noisy_q2_B = laplace_mechanism(true_q2, sensitivity, epsilons_B['q2'])
    noisy_q3_B = laplace_mechanism(true_q3, sensitivity, epsilons_B['q3'])
    
    print(f"Q1 加噪结果: {noisy_q1_B:.2f} (误差: {abs(noisy_q1_B - true_q1):.2f})")
    print(f"Q2 加噪结果: {noisy_q2_B:.2f} (误差: {abs(noisy_q2_B - true_q2):.2f})")
    print(f"Q3 加噪结果: {noisy_q3_B:.2f} (误差: {abs(noisy_q3_B - true_q3):.2f})")

if __name__ == "__main__":
    main()
# dp_utils.py
import numpy as np

def laplace_mechanism(query_result, sensitivity, epsilon):
    """
    向查询结果中添加拉普拉斯噪声以实现中心化差分隐私。

    参数:
    query_result (float): 原始的、未加噪的查询结果。
    sensitivity (float): 查询的敏感度。
    epsilon (float): 隐私预算。

    返回:
    float: 添加噪声后的查询结果。
    """
    if epsilon <= 0:
        raise ValueError("Epsilon必须为正数。")
        
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=1)
    return query_result + noise[0]

def randomized_response(true_value, epsilon):
    """
    对单个二元数据点应用随机响应机制以实现本地化差分隐私。

    参数:
    true_value (int): 用户的真实值 (0 或 1)。
    epsilon (float): 隐私预算。

    返回:
    int: 扰动后的值 (0 或 1)。
    """
    if epsilon <= 0:
        raise ValueError("Epsilon必须为正数。")

    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    
    if np.random.random() < p:
        return true_value
    else:
        return 1 - true_value

def correct_result(perturbed_data, epsilon):
    """
    对经过随机响应扰动的数据进行统计和校正，以估算真实的正向案例数。

    参数:
    perturbed_data (list or np.array): 包含所有用户扰动后数据的列表。
    epsilon (float): 用于扰动的隐私预算。

    返回:
    float: 对真实正向案例数量的估算值。
    """
    if epsilon <= 0:
        raise ValueError("Epsilon必须为正数。")

    N = len(perturbed_data)
    n_prime_yes = sum(perturbed_data)
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    
    if p == 0.5:
        return n_prime_yes
        
    estimated_yes = (n_prime_yes - N * (1 - p)) / (2 * p - 1)
    return estimated_yes

def unary_encoding_perturb(true_choice, K, epsilon):
    """
    对单个多元数据点应用一元编码和随机响应。

    参数:
    true_choice (int): 用户的真实选项 (0 到 K-1)。
    K (int): 选项总数。
    epsilon (float): 隐私预算。

    返回:
    np.array: 扰动后的一元编码向量。
    """
    # 编码：创建one-hot向量
    one_hot_vector = np.zeros(K)
    one_hot_vector[true_choice] = 1

    # 扰动：对向量的每一位应用随机响应
    perturbed_vector = np.array([randomized_response(bit, epsilon) for bit in one_hot_vector])

    return perturbed_vector

def unary_decoding_correct(perturbed_vectors, epsilon):
    """
    对收集到的、经过一元编码扰动的数据进行聚合与校正。

    参数:
    perturbed_vectors (np.array): 一个N x K的数组,N是用户数,K是选项数。
    epsilon (float): 隐私预算。

    返回:
    np.array: 估算的K个选项的真实计票数。
    """
    N, K = perturbed_vectors.shape

    # 1. 聚合：按列求和，得到每个选项被报告为1的总次数
    aggregated_counts = np.sum(perturbed_vectors, axis=0)

    # 2. 校正：对每个选项的聚合结果应用校正公式
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)

    if p == 0.5:
        # 无法有效校正
        return aggregated_counts

    estimated_counts = (aggregated_counts - N * (1 - p)) / (2 * p - 1)
    return estimated_counts
# 差分隐私技术原理与应用探索实验

## 概述
本项目为一系列差分隐私（Differential Privacy, DP）编程实验，旨在系统学习、实现并评估差分隐私技术在统计与机器学习场景中的应用。实验从基础机制到实战训练覆盖五个核心模块，便于复现实验与比较隐私-效用权衡。

## 目录
- 项目简介
- 实验模块一览
- 环境与依赖
- 文件结构说明
- 快速运行指南
- 结果预期

## 实验模块一览
1. CDP 基础应用：中心化差分隐私（拉普拉斯机制）对计数查询加噪。  
2. LDP 参数分析：本地差分隐私（随机响应）中隐私预算 ε 与用户数 N 的影响分析。  
3. LDP 进阶应用：一元编码（Unary Encoding）在高基数频率估计中的应用。  
4. 差分隐私机器学习：使用 Opacus 的 DP-SGD 在 MNIST 上训练并绘制隐私-准确率曲线。  
5. 隐私预算管理：模拟多查询场景，比较不同预算分配策略的效果（组合性验证）。

## 环境与依赖
建议使用 Conda 管理 Python 环境（推荐 Python 3.9）。

创建并激活环境：
```bash
conda create -n privacy_lab python=3.9 -y
conda activate privacy_lab
```

安装依赖：
```bash
conda install numpy matplotlib pandas tqdm -y
pip install torch torchvision opacus
```

## 文件结构说明
- dp_utils.py  
  功能：实现项目所需的差分隐私机制（拉普拉斯机制、随机响应、LDP 校正、一元编码等）。
- run_basic_exp.py  
  功能：实验一（CDP 基础应用）。
- run_ldp_analysis.py  
  功能：实验二（LDP 参数分析：ε 与 N 的影响）。
- run_unary_ldp.py  
  功能：实验三（LDP 在多类别/高基数数据的频率估计）。
- run_dp_mnist_basic.py  
  功能：实验四基线（非隐私）模型训练。
- run_dp_mnist.py  
  功能：实验四（使用 Opacus 的 DP-SGD）。
- plot_results.py  
  功能：绘制实验四的“隐私—准确率”权衡曲线。
- run_composition_exp.py  
  功能：实验五（隐私预算分配与组合性实验）。

## 快速运行指南
请在激活的 privacy_lab 环境下执行下列命令。

实验一：CDP 基础应用
```bash
python run_basic_exp.py
```

实验二：LDP 参数分析
```bash
python run_ldp_analysis.py
```

实验三：LDP 在高基数场景
```bash
python run_unary_ldp.py
```

实验四：差分隐私机器学习（建议三步）
1. 训练非隐私基线：
```bash
python run_dp_mnist_basic.py
```
2. 训练差分隐私模型（可在脚本内修改 target_epsilon）：
```bash
python run_dp_mnist.py
```
3. 可视化多次实验结果（在 plot_results.py 顶部填入结果数据）：
```bash
python plot_results.py
```

实验五：隐私预算管理（多查询组合实验）
```bash
python run_composition_exp.py
```

## 预期输出
- run_basic_exp.py：显示真实计数与多次拉普拉斯噪声后的输出与误差。  
- run_ldp_analysis.py：输出 MAE / MRE 并绘制随 ε 与 N 变化的曲线。  
- run_unary_ldp.py：输出真实与 LDP 估计的类别计数、MAE，并绘制条形对比图。  
- run_dp_mnist_basic.py：打印基线训练日志与测试准确率。  
- run_dp_mnist.py：打印 DP-SGD 训练日志、每 epoch 的测试准确率及当前 (ε, δ)。  
- plot_results.py：绘制隐私—准确率权衡图。  
- run_composition_exp.py：打印多查询真实值和不同预算分配下的加噪结果与误差对比。

Aero-Engine Remaining Useful Life (RUL) Prediction System
基于深度学习（CNN-LSTM-Attention）的航空发动机剩余寿命预测研究

1. 项目简介 (Project Overview)
本项目针对航空发动机在复杂多工况下的非线性退化难题，基于典型的 C-MAPSS 仿真数据集，实现了从基础时序模型到时空特征融合模型的演进。通过对比三种主流深度学习架构，深入探讨了不同模型对发动机退化特征的提取能力。

2. 模型架构演进 (Model Evolution)
本项目包含三个核心模型实现，展示了算法的迭代过程：

Base Model (LSTM): 利用双层 LSTM 捕获传感器数据的长程时间依赖特性。

Enhanced Model (1D-CNN + LSTM): 引入一维卷积层提取局部空间特征，结合 LSTM 处理全局退化趋势。

Advanced Model (CNN-LSTM-Attention): 在前两者的基础上集成 自注意力机制 (Self-Attention)，实现对临近失效期关键信号的自适应聚焦，显著提升预警精度。

3. 核心技术点 (Technical Highlights)
数据预处理： 使用 Pandas/NumPy 对 13 路关键传感器特征进行筛选。

信号降噪： 应用 滑动平均算法 剔除高频环境白噪声，提取单调退化特征轮廓。

标签工程： 构建 分段线性 (Piecewise Linear) 标签模型，有效解决发动机健康早期学习效率低下的问题。

优化策略： 针对单一工况过拟合现象，引入正则化与动态学习率调优，提升模型泛化性能。

4. 实验结果 (Experimental Results)
4.1 预测性能对比
模型在测试集上表现优异，尤其是在 RUL < 40 的临近失效阶段，预测趋势与真实值高度拟合。

<img width="572" height="265" alt="image" src="https://github.com/user-attachments/assets/7893b962-9a81-418b-80b1-254b7dbe9d2b" />


4.2 训练过程监控
收敛性： 模型在第 13 个 Epoch 出现明显梯度下降，MSE Loss 快速收敛至理想区间。

指标分析： 详细对比了各模型的 MASE、MSE 等关键评价指标。

5. 快速启动 (Getting Started)
环境要求
Python 3.x

PyTorch

Pandas / NumPy / Matplotlib

运行指南 (Usage)
数据准备： 请确保将 C-MAPSS 数据集解压并放置在项目根目录的 archive/ 文件夹下。

运行不同版本的模型：

运行基础 LSTM 模型：

Bash
python 001train_LSTM.py
运行 CNN-LSTM 特征融合模型：

Bash
python 001train_cnn_lstm.py
运行最终的注意力机制模型：

Bash
python 001train_lstm_cnn_attention.py

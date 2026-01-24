# 异常检测论文总结

本目录包含工业异常检测领域的核心论文。本文档总结了每篇论文的整体解决思路，并按照主要范式进行分类，最后按照演进路线进行排序。

## 论文总结

### 1. PaDiM - a Patch Distribution Modeling Framework (2020)
**解决思路**：对每个图像patch建立多元高斯分布模型，通过预训练CNN提取特征，计算patch特征的均值和协方差矩阵，异常检测时计算测试样本与正常分布的马氏距离。

### 2. DRÆM - A discriminatively trained reconstruction embedding for surface anomaly detection (2021)
**解决思路**：使用判别式训练的重建嵌入，通过重建正常样本和合成异常样本，训练一个能够区分正常和异常的重建模型，利用重建误差进行异常检测。

### 3. FastFlow - Unsupervised Anomaly Detection and Localization (2021)
**解决思路**：使用2D归一化流（Normalizing Flow）作为概率分布估计器，在预训练特征图上学习正常数据的分布，通过计算似然值进行异常检测和定位。

### 4. PatchCore - Towards Total Recall in Industrial Anomaly Detection (2022)
**解决思路**：构建核心patch记忆库，通过贪心算法选择最具代表性的patch特征，使用最近邻搜索进行异常检测，在保持高召回率的同时提升检测效率。

### 5. CFA - Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization (2022)
**解决思路**：使用耦合超球面进行特征适应，将正常样本特征映射到超球面中心，异常样本偏离中心，通过计算特征到超球面中心的距离进行异常定位。

### 6. RD - Anomaly Detection via Reverse Distillation from One-Class Embedding (2022)
**解决思路**：采用反向蒸馏架构，教师网络学习正常样本的单类嵌入，学生网络通过反向蒸馏从教师网络学习，利用知识蒸馏的差异进行异常检测。

### 7. SoftPatch - Unsupervised Anomaly Detection with Noisy Data (2022)
**解决思路**：针对噪声数据，使用软patch选择机制，通过评估patch的质量和可靠性，动态调整patch权重，减少噪声数据对异常检测的影响。

### 8. SimpleNet - A Simple Network for Image Anomaly Detection and Localization (2023)
**解决思路**：使用简单的特征提取器提取多尺度特征，通过浅层判别器网络区分正常和异常特征，采用特征适配器和异常特征生成器提升检测性能。

### 9. WinCLIP - Zero-Few-Shot Anomaly Classification and Segmentation (2023)
**解决思路**：利用CLIP的视觉-语言对齐能力，通过文本提示和图像特征匹配，实现零样本和少样本的异常分类和分割，无需训练即可适应新类别。

### 10. Dinomaly - The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection (2024)
**解决思路**：采用"少即是多"的哲学，通过简化模型架构和训练过程，使用轻量级特征提取和简单的异常评分机制，实现多类无监督异常检测。

### 11. DiAD - A Diffusion-based Framework for Multi-class Anomaly Detection (2024)
**解决思路**：使用扩散模型学习正常数据的分布，通过去噪过程重建图像，利用重建误差和扩散过程的异常分数进行多类异常检测。

## 范式分类

### 1. 基于特征分布建模的方法
- **PaDiM (2020)**: Patch多元高斯分布建模
- **PatchCore (2022)**: 核心patch记忆库
- **CFA (2022)**: 耦合超球面特征适应

### 2. 基于生成模型的方法
- **FastFlow (2021)**: 归一化流概率分布估计
- **DRÆM (2021)**: 判别式重建嵌入
- **DiAD (2024)**: 扩散模型

### 3. 基于知识蒸馏的方法
- **RD (2022)**: 反向蒸馏

### 4. 基于判别器的方法
- **SimpleNet (2023)**: 特征提取+判别器

### 5. 基于预训练模型的方法
- **WinCLIP (2023)**: CLIP零样本/少样本

### 6. 改进/变体方法
- **SoftPatch (2022)**: 噪声数据处理的patch方法
- **Dinomaly (2024)**: 轻量级多类异常检测

## 演进路线

### 第一阶段：基础特征分布建模 (2020)
1. **PaDiM**: 开创性地使用patch级别的多元高斯分布建模，奠定了基于特征分布的方法基础

### 第二阶段：生成模型与重建方法 (2021)
2. **DRÆM**: 引入判别式训练的重建方法，结合合成异常样本提升检测能力
3. **FastFlow**: 使用归一化流进行概率分布估计，提供可解释的异常分数

### 第三阶段：记忆库与特征适应 (2022)
4. **PatchCore**: 通过核心patch选择优化记忆库，提升效率和召回率
5. **CFA**: 使用超球面进行特征适应，简化异常检测为距离计算
6. **RD**: 引入反向蒸馏架构，利用知识蒸馏进行异常检测
7. **SoftPatch**: 针对噪声数据场景，提出软patch选择机制

### 第四阶段：简化与预训练模型 (2023)
8. **SimpleNet**: 简化网络架构，使用浅层判别器实现高效检测
9. **WinCLIP**: 利用大规模预训练CLIP模型，实现零样本/少样本异常检测

### 第五阶段：轻量级与扩散模型 (2024)
10. **Dinomaly**: 采用"少即是多"哲学，简化模型实现多类检测
11. **DiAD**: 结合扩散模型，利用生成模型的强大能力进行异常检测

## 演进趋势总结

1. **从复杂到简化**：早期方法（PaDiM）建立复杂分布模型，后期方法（SimpleNet, Dinomaly）追求简洁高效
2. **从单一到多样**：从单一范式（分布建模）发展到多种范式（生成模型、知识蒸馏、预训练模型）
3. **从有监督到无监督**：逐步减少对异常样本的依赖，向完全无监督方向发展
4. **从特定到通用**：从单类检测发展到多类检测，从需要训练到零样本/少样本
5. **从传统到前沿**：从传统CNN特征到预训练模型（CLIP）和生成模型（扩散模型）


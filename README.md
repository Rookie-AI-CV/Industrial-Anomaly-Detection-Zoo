# Industrial Anomaly Detection Zoo

本资源库系统整理了工业异常与缺陷检测领域的研究论文、代码实现及配套的Jupyter notebooks。内容涵盖基于重建的方法、生成式方法、分类方法、检测方法以及分割方法等主要技术路线。

**作者**: Rookie  
**邮箱**: RookieEmail@163.com

## 目录结构

```
Industrial-Anomaly-Detection-Zoo/
├── papers/                    # 研究论文与文献
│   ├── survey/                # 综述论文
│   ├── reconstruction/        # 基于重建的方法
│   │   └── references/        # 相关引用文献
│   ├── generation/            # 生成式方法
│   │   └── references/        # 相关引用文献
│   ├── classification/        # 分类方法
│   │   └── references/        # 相关引用文献
│   ├── detection/             # 检测方法
│   │   └── references/        # 相关引用文献
│   └── segmentation/          # 分割方法
│       └── references/        # 相关引用文献
├── code/                      # 代码实现
│   ├── reconstruction/        # 基于重建的方法实现
│   ├── generation/            # 生成式方法实现
│   ├── classification/        # 分类方法实现
│   ├── detection/             # 检测方法实现
│   └── segmentation/          # 分割方法实现
├── notebooks/                 # Jupyter notebooks
│   ├── reconstruction/        # 重建方法相关notebooks
│   ├── generation/            # 生成方法相关notebooks
│   ├── classification/        # 分类方法相关notebooks
│   ├── detection/             # 检测方法相关notebooks
│   └── segmentation/          # 分割方法相关notebooks
├── datasets/                  # 数据集目录
├── results/                   # 实验结果
└── requirements.txt           # 依赖包列表
```

## 方法分类

### 基于重建的方法 (Reconstruction-based Methods)

基于重建的方法通过训练模型学习正常样本的表示，异常样本在重建时会产生较大的误差。主要包括：

- 自编码器 (Autoencoder)
- 变分自编码器 (Variational Autoencoder, VAE)
- 生成对抗网络 (Generative Adversarial Networks, GANs)
- 其他基于重建的异常检测方法

### 生成式方法 (Generation Methods)

生成式方法通过生成模型学习数据分布，利用生成模型对异常样本的判别能力进行异常检测。主要包括：

- 生成对抗网络 (GANs)
- 变分自编码器 (VAE)
- 扩散模型 (Diffusion Models)
- 其他生成式异常检测方法

### 分类方法 (Classification Methods)

基于分类的方法通过训练分类器区分正常与异常样本。主要包括：

- 监督学习方法
- 半监督学习方法
- 少样本学习方法

### 检测方法 (Detection Methods)

检测方法专注于定位图像或数据中的异常区域。主要包括：

- 目标检测方法
- 异常检测方法
- 其他检测方法

### 分割方法 (Segmentation Methods)

分割方法旨在对异常区域进行像素级或区域级的精确分割。主要包括：

- 语义分割
- 实例分割
- 异常区域分割

## 环境配置

安装所需依赖包：

```bash
pip install -r requirements.txt
```

## 数据集

数据集需单独下载，请参考各数据集的官方说明文档。

## 论文资源

本资源库按方法类别组织论文资源，每个方法类别目录下包含：

- 主要研究论文：该类别下的核心论文
- 引用文献 (`references/`): 相关论文中引用的重要文献，便于追溯方法的理论基础

综述论文统一存放在 `papers/survey/` 目录下。

## 贡献

欢迎提交代码实现、论文资源及相关notebooks。请确保提交的内容符合学术规范。

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

MIT License 允许您自由使用、修改、分发本项目的代码和资源，但需保留原始许可证声明和版权信息。使用本项目的论文、代码或资源时，请遵循以下原则：

1. 保留原始版权声明和许可证文件
2. 引用相关论文时请遵循学术规范，正确标注来源
3. 论文资源的使用需遵守原论文的版权要求
4. 数据集的使用需遵守相应数据集的许可协议

对于资源库中引用的论文、代码和数据集，其版权归原作者所有。本资源库仅作为学术研究用途的整理和索引，不拥有这些资源的版权。

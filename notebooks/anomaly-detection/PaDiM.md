# PaDiM - Patch Distribution Modeling Framework 完整算法解读

## 论文信息
- **标题**: PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
- **年份**: 2020
- **论文链接**: https://arxiv.org/abs/2011.08785
- **核心思想**: 对每个图像patch建立多元高斯分布模型进行异常检测

## 算法概述

PaDiM是一种基于patch分布建模的异常检测方法。它利用预训练CNN提取多尺度特征，对每个空间位置的patch建立多元高斯分布，通过计算测试patch与正常分布的马氏距离来检测异常。

### 整体架构

```
输入图像 → 预训练CNN特征提取 → 多尺度特征融合 → 随机降维 → 多元高斯分布建模 → 马氏距离计算 → 异常图生成
```

## 一、特征提取阶段

### 1.1 预训练CNN特征提取器

**代码来源**：`anomalib/models/components/feature_extractors/timm.py` (第42-200行)

PaDiM使用`TimmFeatureExtractor`从预训练CNN的多个中间层提取特征。该组件支持timm库中的各种模型架构。

```python
from anomalib.models.components import TimmFeatureExtractor

# 初始化特征提取器
self.feature_extractor = TimmFeatureExtractor(
    backbone="resnet18",  # 或 "wide_resnet50_2"
    layers=["layer1", "layer2", "layer3"],
    pre_trained=True,
    requires_grad=False  # 冻结参数，不计算梯度
).eval()

# 提取特征
features = self.feature_extractor(input_tensor)
# 返回字典: {'layer1': tensor, 'layer2': tensor, 'layer3': tensor}
```

**实现细节**：
- 使用timm库的`create_model`创建模型，设置`features_only=True`只返回特征图
- 通过`out_indices`指定要提取的层索引
- 特征提取器默认设置为`eval()`模式，不计算梯度

### 1.2 多尺度特征融合

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第198-221行)

将不同层的特征上采样到相同尺寸并拼接，形成多尺度特征表示。

```python
def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
    """Generate embedding from hierarchical feature map.
    
    This method combines features from multiple layers of the backbone network
    to create a rich embedding that captures both low-level and high-level
    image features.
    
    Args:
        features: Dictionary mapping layer names to their feature tensors
        
    Returns:
        Embedding tensor with reduced dimensions
    """
    # 从第一个层开始（通常分辨率最高）
    embeddings = features[self.layers[0]]
    
    # 拼接其他层的特征（上采样到相同尺寸）
    for layer in self.layers[1:]:
        layer_embedding = features[layer]
        # 使用nearest模式上采样，保持空间结构
        layer_embedding = F.interpolate(
            layer_embedding, 
            size=embeddings.shape[-2:], 
            mode="nearest"
        )
        embeddings = torch.cat((embeddings, layer_embedding), 1)
    
    # 使用随机索引进行降维
    idx = self.idx.to(embeddings.device)
    return torch.index_select(embeddings, 1, idx)
```

**关键设计**：
- 使用`nearest`模式上采样而非`bilinear`，保持特征的离散性
- 直接拼接不同层的特征，不进行额外的融合操作
- 通过`torch.index_select`高效选择特征维度，避免复制整个特征图

## 二、训练阶段（分布建模）

### 2.1 模型初始化与特征降维

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第110-150行)

PaDiM在初始化时随机选择特征维度进行降维，这是论文中的关键优化。

```python
class PadimModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None
        
        self.backbone = backbone
        self.layers = layers
        
        # 特征提取器
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=layers,
            pre_trained=pre_trained,
        ).eval()
        
        # 计算原始特征维度（所有层通道数之和）
        self.n_features_original = sum(self.feature_extractor.out_dims)
        
        # 降维后的特征数（论文默认值：resnet18=100, wide_resnet50_2=550）
        self.n_features = n_features or _N_FEATURES_DEFAULTS.get(self.backbone)
        
        # 随机选择特征维度索引（保存为buffer，确保可复现）
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(self.n_features_original), self.n_features)),
        )
        self.idx: torch.Tensor
        
        # 多元高斯分布建模组件
        self.gaussian = MultiVariateGaussian()
        self.anomaly_map_generator = AnomalyMapGenerator()
        
        # 训练时收集特征的memory bank
        self.memory_bank: list[torch.tensor] = []
```

**关键设计**：
- 使用`register_buffer`保存随机索引，确保模型保存/加载后结果一致
- 随机降维是PaDiM的核心创新，大幅减少计算量而不显著影响性能
- `MultiVariateGaussian`负责计算每个patch位置的均值和协方差

### 2.2 训练时的特征收集

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第152-196行)

训练时，模型前向传播收集所有正常样本的特征到memory bank。

```python
def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
    """Forward-pass image-batch (N, C, H, W) into model to extract features."""
    output_size = input_tensor.shape[-2:]
    
    # 支持大图像的分块处理
    if self.tiler:
        input_tensor = self.tiler.tile(input_tensor)
    
    # 特征提取（冻结梯度，不计算反向传播）
    with torch.no_grad():
        features = self.feature_extractor(input_tensor)
        embeddings = self.generate_embedding(features)
    
    if self.tiler:
        embeddings = self.tiler.untile(embeddings)
    
    if self.training:
        # 训练时：收集特征到memory bank
        self.memory_bank.append(embeddings)
        return embeddings
    
    # 推理时：计算异常分数（见后续章节）
    anomaly_map = self.anomaly_map_generator(
        embedding=embeddings,
        mean=self.gaussian.mean,
        inv_covariance=self.gaussian.inv_covariance,
        image_size=output_size,
    )
    pred_score = torch.amax(anomaly_map, dim=(-2, -1))
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
```

**训练流程**（Lightning模型）：
```python
# 代码来源：anomalib/models/image/padim/lightning_model.py (第139-157行)
def training_step(self, batch: Batch, *args, **kwargs) -> None:
    """训练步骤：收集特征"""
    _ = self.model(batch.image)  # 特征被收集到memory_bank
    return torch.tensor(0.0, requires_grad=True, device=self.device)  # 虚拟损失

def fit(self) -> None:
    """训练结束后，拟合高斯分布"""
    logger.info("Fitting a Gaussian to the embedding collected from the training set.")
    self.model.fit()
```

### 2.3 多元高斯分布拟合

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第223-242行)

训练结束后，对memory bank中的所有特征拟合多元高斯分布。

```python
def fit(self) -> None:
    """Fits a Gaussian model to the current contents of the memory bank.
    
    This method is typically called after the memory bank has been filled during training.
    After fitting, the memory bank is cleared to free GPU memory.
    
    Raises:
        ValueError: If the memory bank is empty.
    """
    if len(self.memory_bank) == 0:
        msg = "Memory bank is empty. Cannot perform coreset selection."
        raise ValueError(msg)
    
    # 合并所有训练特征: [B1, C, H, W] + [B2, C, H, W] + ... -> [N, C, H, W]
    self.memory_bank = torch.vstack(self.memory_bank)
    
    # 拟合高斯分布（计算每个patch位置的均值和协方差）
    self.gaussian.fit(self.memory_bank)
    
    # 清空memory bank，释放GPU内存
    self.memory_bank = []
```

**MultiVariateGaussian实现**：

**代码来源**：`anomalib/models/components/stats/multi_variate_gaussian.py` (第128-180行)

```python
class MultiVariateGaussian(DynamicBufferMixin, nn.Module):
    """多元高斯分布建模"""
    
    def __init__(self) -> None:
        super().__init__()
        # 使用register_buffer存储分布参数（支持模型保存/加载）
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_covariance", torch.empty(0))
    
    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """计算多元高斯分布参数：均值和逆协方差矩阵
        
        Args:
            embedding: 输入特征 [B, C, H, W]
                B: batch size (训练样本数)
                C: 特征维度 (降维后的维度，如100)
                H, W: 特征图空间尺寸
        
        Returns:
            [mean, inv_covariance]
            - mean: [C, H*W] - 每个patch位置的均值向量
            - inv_covariance: [H*W, C, C] - 每个patch位置的逆协方差矩阵
        """
        device = embedding.device
        batch, channel, height, width = embedding.size()
        
        # 重塑为 [B, C, H*W]
        embedding_vectors = embedding.view(batch, channel, height * width)
        
        # 计算均值：对batch维度求平均
        self.mean = torch.mean(embedding_vectors, dim=0)  # [C, H*W]
        
        # 为每个patch位置计算协方差矩阵
        covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        
        for i in range(height * width):
            # 提取该位置所有样本的特征 [B, C]
            patch_features = embedding_vectors[:, :, i]
            
            # 计算协方差矩阵（使用_cov方法，类似numpy.cov）
            covariance[:, :, i] = self._cov(patch_features, rowvar=False) + 0.01 * identity
        
        # 转置为 [H*W, C, C] 并添加正则化项
        stabilized_covariance = covariance.permute(2, 0, 1) + 1e-5 * identity
        
        # 计算逆协方差矩阵（预计算，提升推理速度）
        if device.type == "mps":
            # MPS设备需要转到CPU计算
            self.inv_covariance = torch.linalg.inv(stabilized_covariance.cpu()).to(device)
        else:
            self.inv_covariance = torch.linalg.inv(stabilized_covariance)
        
        return [self.mean, self.inv_covariance]
    
    def fit(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """拟合分布（调用forward方法）"""
        return self.forward(embedding)
```

**协方差计算**：

**代码来源**：`anomalib/models/components/stats/multi_variate_gaussian.py` (第57-126行)

```python
@staticmethod
def _cov(
    observations: torch.Tensor,
    rowvar: bool = False,
    bias: bool = False,
    ddof: int | None = None,
    aweights: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算协方差矩阵（类似numpy.cov的实现）
    
    Args:
        observations: [N, D] - N个样本，D维特征
        rowvar: 如果True，每行是一个变量；False则每列是一个变量
        bias: 是否使用有偏估计
        ddof: 自由度调整
        aweights: 样本权重
    
    Returns:
        [D, D] - 协方差矩阵
    """
    # 确保至少2维
    if observations.dim() == 1:
        observations = observations.view(-1, 1)
    
    # 转置（如果需要）
    if rowvar and observations.shape[0] != 1:
        observations = observations.t()
    
    # 计算均值
    if ddof is None:
        ddof = 1 if bias == 0 else 0
    
    if aweights is not None:
        weights_sum = torch.sum(aweights)
        avg = torch.sum(observations * (aweights / weights_sum)[:, None], 0)
    else:
        avg = torch.mean(observations, 0)
    
    # 计算归一化因子
    if weights is None:
        fact = observations.shape[0] - ddof
    # ... (其他情况处理)
    
    # 中心化
    observations_m = observations.sub(avg.expand_as(observations))
    
    # 计算协方差: E[(X - μ)(X - μ)^T]
    x_transposed = observations_m.t()
    covariance = torch.mm(x_transposed, observations_m)
    covariance = covariance / fact
    
    return covariance.squeeze()
```

**关键实现细节**：
- 对每个patch位置独立计算协方差矩阵，共H*W个矩阵
- 使用0.01的权重添加单位矩阵正则化，避免矩阵奇异
- 额外添加1e-5的正则化项，进一步提升数值稳定性
- 预计算逆协方差矩阵，避免推理时重复求逆

## 三、测试阶段（异常检测）

### 3.1 马氏距离计算

**代码来源**：`anomalib/models/image/padim/anomaly_map.py` (第69-95行)

PaDiM使用马氏距离衡量测试patch与正常分布的差异。马氏距离考虑了特征之间的相关性，比欧氏距离更适合高维特征。

**数学公式**（论文Equation 2）：
```
d_M(x, μ, Σ) = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]
```

其中：
- x: 测试patch的特征向量 [C]
- μ: 正常分布的均值向量 [C]
- Σ: 正常分布的协方差矩阵 [C, C]
- Σ⁻¹: 逆协方差矩阵（预计算）

```python
@staticmethod
def compute_distance(embedding: torch.Tensor, stats: list[torch.Tensor]) -> torch.Tensor:
    """使用马氏距离计算异常分数
    
    Implements Equation (2) from Section III-C of the PaDiM paper.
    
    Args:
        embedding: 测试图像的特征嵌入 [B, C, H, W]
        stats: [mean, inv_covariance]
            - mean: [C, H*W] - 每个patch位置的均值
            - inv_covariance: [H*W, C, C] - 每个patch位置的逆协方差矩阵
    
    Returns:
        异常分数图 [B, 1, H, W]
    """
    batch, channel, height, width = embedding.shape
    
    # 重塑为 [B, C, H*W]
    embedding = embedding.reshape(batch, channel, height * width)
    
    mean, inv_covariance = stats
    
    # 计算差值: (x - μ) for each patch
    # embedding: [B, C, H*W], mean: [C, H*W]
    # delta: [H*W, B, C] (转置以便批量计算)
    delta = (embedding - mean).permute(2, 0, 1)
    
    # 计算马氏距离: (x - μ)ᵀ Σ⁻¹ (x - μ)
    # delta: [H*W, B, C]
    # inv_covariance: [H*W, C, C]
    # 对每个patch位置: [B, C] @ [C, C] @ [B, C] -> [B]
    distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
    
    # 重塑为空间图并确保非负
    distances = distances.reshape(batch, 1, height, width)
    return distances.clamp(0).sqrt()
```

**实现优化**：
- 批量计算所有patch的马氏距离，避免循环
- 使用`clamp(0)`确保距离非负（处理数值误差）
- 预计算逆协方差矩阵，避免每次求逆

### 3.2 异常图生成与后处理

**代码来源**：`anomalib/models/image/padim/anomaly_map.py` (第97-157行)

```python
class AnomalyMapGenerator(nn.Module):
    """生成异常热力图"""
    
    def __init__(self, sigma: int = 4) -> None:
        super().__init__()
        # 高斯模糊核大小：2 * int(4.0 * sigma + 0.5) + 1
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            channels=1
        )
    
    @staticmethod
    def up_sample(
        distance: torch.Tensor, 
        image_size: tuple[int, int] | torch.Size
    ) -> torch.Tensor:
        """上采样异常分数到原始图像尺寸"""
        return F.interpolate(
            distance,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
    
    def smooth_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """应用高斯模糊平滑异常图"""
        return self.blur(anomaly_map)
    
    def compute_anomaly_map(
        self,
        embedding: torch.Tensor,
        mean: torch.Tensor,
        inv_covariance: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """计算完整的异常图
        
        流程：
        1. 计算马氏距离
        2. 上采样到原始图像尺寸
        3. 高斯模糊平滑
        """
        # 1. 计算马氏距离
        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        
        # 2. 上采样
        if image_size:
            score_map = self.up_sample(score_map, image_size)
        
        # 3. 高斯模糊平滑
        return self.smooth_anomaly_map(score_map)
```

**后处理说明**：
- **上采样**：使用双线性插值将patch级分数映射到像素级
- **高斯模糊**：平滑异常图，减少噪声和伪影，sigma=4是论文默认值
- **图像级分数**：使用`torch.amax`取异常图的最大值作为图像级异常分数

## 四、关键实现细节与优化

### 4.1 随机特征降维

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第140-144行)

PaDiM的核心创新是随机特征降维，大幅减少计算量。

```python
# 随机选择特征维度索引
self.register_buffer(
    "idx",
    torch.tensor(sample(range(self.n_features_original), self.n_features)),
)

# 在generate_embedding中使用
idx = self.idx.to(embeddings.device)
return torch.index_select(embeddings, 1, idx)
```

**论文分析**：
- 论文发现随机选择100-550维特征即可保持检测性能
- 降维后协方差矩阵从[C, C]变为[reduced_C, reduced_C]，计算量大幅减少
- 使用`register_buffer`保存索引，确保模型可复现

### 4.2 数值稳定性优化

**代码来源**：`anomalib/models/components/stats/multi_variate_gaussian.py` (第149-163行)

```python
# 1. 协方差矩阵正则化（避免奇异）
for i in range(height * width):
    covariance[:, :, i] = self._cov(...) + 0.01 * identity

# 2. 额外稳定性项
stabilized_covariance = covariance.permute(2, 0, 1) + 1e-5 * identity

# 3. 安全的矩阵求逆（处理MPS设备）
if device.type == "mps":
    self.inv_covariance = torch.linalg.inv(stabilized_covariance.cpu()).to(device)
else:
    self.inv_covariance = torch.linalg.inv(stabilized_covariance)
```

**优化说明**：
- **双重正则化**：0.01权重的主要正则化 + 1e-5的稳定性项
- **设备兼容性**：MPS设备需要转到CPU求逆
- **预计算逆矩阵**：避免推理时重复求逆，提升速度

### 4.3 内存管理

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第223-242行)

```python
def fit(self) -> None:
    # 合并所有特征
    self.memory_bank = torch.vstack(self.memory_bank)
    
    # 拟合分布
    self.gaussian.fit(self.memory_bank)
    
    # 清空memory bank释放GPU内存
    self.memory_bank = []
```

**设计考虑**：
- 训练时逐batch收集特征，避免一次性加载所有数据
- 拟合后立即清空memory bank，释放GPU内存
- 分布参数存储在buffer中，随模型保存/加载

### 4.4 大图像处理（Tiler支持）

**代码来源**：`anomalib/models/image/padim/torch_model.py` (第175-183行)

```python
if self.tiler:
    input_tensor = self.tiler.tile(input_tensor)  # 分块

with torch.no_grad():
    features = self.feature_extractor(input_tensor)
    embeddings = self.generate_embedding(features)

if self.tiler:
    embeddings = self.tiler.untile(embeddings)  # 合并
```

支持大图像的分块处理，避免内存溢出。

## 五、完整训练与推理流程

### 5.1 训练流程

**代码来源**：`anomalib/models/image/padim/lightning_model.py`

```python
# 1. 初始化模型
model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
    pre_trained=True,
    n_features=100
)

# 2. 训练阶段（收集特征）
def training_step(self, batch: Batch, *args, **kwargs) -> None:
    _ = self.model(batch.image)  # 特征收集到memory_bank
    return torch.tensor(0.0, requires_grad=True, device=self.device)

# 3. 训练结束后拟合分布
def fit(self) -> None:
    self.model.fit()  # 计算均值和协方差
```

**流程总结**：
1. 遍历所有正常训练图像
2. 对每张图像提取多尺度特征并降维
3. 收集所有特征到memory bank
4. 训练结束后，对每个patch位置拟合多元高斯分布
5. 保存分布参数（均值、逆协方差）

### 5.2 推理流程

```python
# 1. 加载模型（包含分布参数）
model = Padim.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 对测试图像进行前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

**推理步骤**：
1. 提取测试图像的多尺度特征
2. 对每个patch计算马氏距离
3. 上采样到原始图像尺寸
4. 高斯模糊平滑
5. 返回异常分数和异常图

## 六、算法优缺点分析

### 6.1 优点

1. **无需训练**：直接使用预训练特征，无需反向传播
2. **计算高效**：随机降维大幅减少计算量，推理速度快
3. **可解释性强**：异常分数有明确的统计意义（马氏距离）
4. **内存友好**：训练后清空memory bank，只保存分布参数
5. **实现简单**：代码逻辑清晰，易于理解和实现

### 6.2 缺点

1. **内存占用**：需要存储H*W个协方差矩阵（每个C×C）
2. **分布假设**：假设特征服从高斯分布，可能不符合实际情况
3. **降维随机性**：随机降维可能丢失重要特征维度
4. **固定阈值**：需要手动设置异常阈值，缺乏自适应能力

## 七、关键超参数

### 7.1 模型超参数

- **backbone**：特征提取器（`"resnet18"` 或 `"wide_resnet50_2"`）
- **layers**：提取特征的层（默认`["layer1", "layer2", "layer3"]`）
- **n_features**：降维后的特征数（resnet18=100, wide_resnet50_2=550）
- **pre_trained**：是否使用预训练权重（默认`True`）

### 7.2 后处理超参数

- **sigma**：高斯模糊的标准差（默认4，在`AnomalyMapGenerator`中）

### 7.3 论文推荐配置

根据论文Table 1：
- **ResNet18**：n_features=100，layers=["layer1", "layer2", "layer3"]
- **Wide ResNet-50-2**：n_features=550，layers=["layer2", "layer3"]

## 八、总结

PaDiM是一个简单而有效的异常检测方法，其核心思想是：
1. **多尺度特征融合**：结合不同语义层次的特征
2. **随机降维**：大幅减少计算量而不显著影响性能
3. **分布建模**：对每个patch位置建立多元高斯分布
4. **马氏距离**：考虑特征相关性的距离度量

该方法无需训练，直接使用预训练特征，实现简单，推理快速，是工业异常检测的经典方法之一。


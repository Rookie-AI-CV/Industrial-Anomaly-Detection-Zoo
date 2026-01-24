# PatchCore - Towards Total Recall in Industrial Anomaly Detection 完整算法解读

## 论文信息
- **标题**: Towards Total Recall in Industrial Anomaly Detection
- **年份**: 2022
- **论文链接**: https://arxiv.org/abs/2106.08265
- **核心思想**: 构建核心patch记忆库，通过K-Center Greedy算法选择最具代表性的patch特征，使用最近邻搜索进行异常检测

## 算法概述

PatchCore是PatchCore是PaDiM的改进版本，通过核心集（coreset）选择大幅减少内存占用，同时保持高召回率。核心创新包括：
1. **核心集选择**：使用K-Center Greedy算法选择代表性patch
2. **局部聚合**：使用Adaptive Average Pooling聚合patch特征
3. **高效最近邻搜索**：使用预计算的记忆库进行快速检索

### 整体架构

```
输入图像 → 预训练特征提取 → 特征池化 → 核心集选择 → 记忆库构建 → 最近邻搜索 → 异常分数计算
```

## 一、特征提取与预处理

### 1.1 特征提取

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第53-129行)

PatchCore使用`TimmFeatureExtractor`提取多尺度特征，与PaDiM类似。

```python
from anomalib.models.components import TimmFeatureExtractor

# 初始化特征提取器
self.feature_extractor = TimmFeatureExtractor(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],  # 通常只使用后两层
    pre_trained=True,
    requires_grad=False
).eval()
```

### 1.2 特征池化与融合

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第196-225行)

PatchCore使用Adaptive Average Pooling将特征图池化到固定尺寸，然后拼接。

```python
def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
    """Generate embedding from hierarchical feature map.
    
    Args:
        features: Dictionary mapping layer names to feature tensors
        
    Returns:
        Embedding tensor [B, C, H, W]
    """
    embeddings_list = []
    
    for layer in self.layers:
        layer_embedding = features[layer]
        
        # Adaptive Average Pooling到固定尺寸（通常与原图尺寸相同）
        layer_embedding = self.feature_pooler(layer_embedding)
        
        # 上采样到相同尺寸（如果需要）
        if layer != self.layers[0]:
            layer_embedding = F.interpolate(
                layer_embedding,
                size=embeddings_list[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        embeddings_list.append(layer_embedding)
    
    # 拼接多尺度特征
    embedding = torch.cat(embeddings_list, dim=1)
    
    # L2归一化（可选，取决于配置）
    if self.normalize:
        embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding
```

**关键组件 - FeaturePooler**：

​	**代码来源：**`anomalib/models/image/patchcore/torch_model.py` (第131-145行)

```python
class FeaturePooler(nn.Module):
    """Adaptive Average Pooling to fixed size"""
    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.pooler = nn.AdaptiveAvgPool2d((pool_size, pool_size))
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pooler(features)
```

**设计说明**：
- **Adaptive Pooling**：将不同尺寸的特征图池化到固定尺寸，便于后续处理
- **L2归一化**：可选，归一化后可以使用余弦相似度代替欧氏距离
- **多尺度融合**：拼接不同层的特征，保留多尺度信息

## 二、核心集选择（Coreset Selection）

### 2.1 K-Center Greedy算法

**代码来源**：`anomalib/models/components/sampling/k_center_greedy.py` (第18-149行)

PatchCore使用K-Center Greedy算法选择核心集，目标是选择k个点使得所有点到最近中心的距离最小。

**算法原理**：
1. 随机选择第一个中心点
2. 迭代选择距离已选中心最远的点作为新中心
3. 重复直到选择k个点

```python
class KCenterGreedy:
    """k-center-greedy method for coreset selection.
    
    Returns points that minimizes the maximum distance of any point to a center.
    Reference: https://arxiv.org/abs/1708.00489
    """
    
    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding  # [N, C] - 所有patch特征
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)  # 核心集大小
        
        # 使用稀疏随机投影降维（加速距离计算）
        self.model = SparseRandomProjection(eps=0.9)
        
        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]
    
    def update_distances(self, cluster_center: int | torch.Tensor | None) -> None:
        """更新到最近中心的距离"""
        if cluster_center is not None:
            center = self.features[cluster_center]
            center = center.squeeze()
            
            # 计算到新中心的距离
            distances = torch.linalg.norm(
                self.features - center, 
                ord=2, 
                dim=1, 
                keepdim=True
            )
            
            # 更新最小距离
            if self.min_distances is None:
                self.min_distances = distances
            else:
                self.min_distances = torch.minimum(self.min_distances, distances)
    
    def get_new_idx(self) -> torch.Tensor:
        """选择距离已选中心最远的点"""
        _, idx = torch.max(self.min_distances.squeeze(1), dim=0)
        return idx
    
    def select_coreset_idxs(self) -> list[int]:
        """贪心选择核心集索引"""
        # 降维（如果特征维度高）
        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
        
        self.reset_distances()
        
        # 随机选择第一个中心
        idx = torch.randint(
            high=self.n_observations, 
            size=(1,), 
            device=self.features.device
        ).squeeze()
        
        selected_coreset_idxs: list[int] = []
        for _ in tqdm(range(self.coreset_size), desc="Selecting Coreset Indices."):
            # 更新距离
            self.update_distances(cluster_center=idx)
            
            # 选择最远的点
            idx = self.get_new_idx()
            
            # 将选中点的距离设为0（避免重复选择）
            self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
            
            selected_coreset_idxs.append(int(idx.item()))
        
        return selected_coreset_idxs
    
    def sample_coreset(self) -> torch.Tensor:
        """选择核心集"""
        idxs = self.select_coreset_idxs()
        return self.embedding[idxs]
```

**优化技巧**：
- **稀疏随机投影**：使用`SparseRandomProjection`降维，加速距离计算
- **增量更新**：每次只更新到新中心的距离，使用`torch.minimum`保持最小距离
- **批量计算**：使用`torch.linalg.norm`批量计算距离，避免循环

### 2.2 核心集选择在PatchCore中的应用

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第252-282行)

```python
def subsample_embedding(self, sampling_ratio: float) -> None:
    """使用K-Center Greedy选择核心集
    
    Args:
        sampling_ratio: 核心集大小比例（如0.1表示10%）
    """
    # 重塑为 [N_patches, C]
    embedding = self.reshape_embedding(self.embedding_store)
    
    # 使用K-Center Greedy选择核心集
    sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
    sampled_idx = sampler.select_coreset_idxs()
    
    # 更新memory bank为选中的核心集
    self.memory_bank = embedding[sampled_idx]
    
    # 清空embedding_store释放内存
    self.embedding_store = []
```

**关键方法 - reshape_embedding**：

```python
def reshape_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
    """将4D特征图重塑为2D矩阵 [N_patches, C]
    
    Args:
        embedding: [B, C, H, W] 或 list of [B, C, H, W]
    
    Returns:
        [N, C] - N是所有patch的总数
    """
    if isinstance(embedding, list):
        embedding = torch.vstack(embedding)
    
    # [B, C, H, W] -> [B*H*W, C]
    batch_size, channels, height, width = embedding.shape
    embedding = embedding.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
    
    return embedding
```

## 三、训练阶段（记忆库构建）

### 3.1 特征收集

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第152-195行)

训练时，PatchCore收集所有正常样本的特征到`embedding_store`。

```python
def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
    """Forward pass"""
    output_size = input_tensor.shape[-2:]
    
    if self.tiler:
        input_tensor = self.tiler.tile(input_tensor)
    
    with torch.no_grad():
        features = self.feature_extractor(input_tensor)
        embeddings = self.generate_embedding(features)
    
    if self.tiler:
        embeddings = self.tiler.untile(embeddings)
    
    if self.training:
        # 训练时：收集特征到embedding_store
        self.embedding_store.append(embeddings)
        return embeddings
    
    # 推理时：最近邻搜索（见后续章节）
    # ...
```

### 3.2 核心集选择与记忆库构建

训练结束后，使用K-Center Greedy选择核心集并构建记忆库。

```python
def subsample_embedding(self, sampling_ratio: float) -> None:
    """使用K-Center Greedy选择核心集并构建记忆库"""
    # 重塑为 [N_patches, C]
    embedding = self.reshape_embedding(self.embedding_store)
    
    # K-Center Greedy选择核心集
    sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
    self.memory_bank = sampler.sample_coreset()
    
    # 清空embedding_store释放内存
    self.embedding_store = []
```

## 四、推理阶段（异常检测）

### 4.1 高效欧氏距离计算

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第284-315行)

PatchCore实现了高效的欧氏距离计算，避免使用`torch.cdist`以支持ONNX导出。

```python
@staticmethod
def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算两组向量之间的成对欧氏距离
    
    使用矩阵运算高效计算，避免torch.cdist以支持ONNX导出。
    
    Args:
        x: [n, d] - 第一组向量
        y: [m, d] - 第二组向量
    
    Returns:
        [n, m] - 距离矩阵，元素(i,j)是x[i]和y[j]的距离
    
    数学公式：
        d(x, y) = √(|x|² - 2xyᵀ + |y|²)
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|²
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|²
    
    # 欧氏距离 = √(|x|² - 2xyᵀ + |y|²)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    
    return res.clamp_min_(0).sqrt_()
```

**优化说明**：
- 使用矩阵乘法代替循环，大幅提升速度
- `clamp_min_(0)`确保距离非负（处理数值误差）
- 避免`torch.cdist`以支持模型导出

### 4.2 最近邻搜索

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第317-347行)

```python
def nearest_neighbors(
    self, 
    embedding: torch.Tensor, 
    n_neighbors: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """在记忆库中查找最近邻
    
    Args:
        embedding: [n, d] - 查询嵌入
        n_neighbors: 返回的最近邻数量
    
    Returns:
        (distances, indices)
        - distances: [n, k] - 到最近邻的距离
        - indices: [n, k] - 最近邻在记忆库中的索引
    """
    # 计算到记忆库的距离
    distances = self.euclidean_dist(embedding, self.memory_bank)
    
    if n_neighbors == 1:
        # 优化：当k=1时使用min代替topk
        patch_scores, locations = distances.min(1)
    else:
        # 查找k个最近邻
        patch_scores, locations = distances.topk(
            k=n_neighbors, 
            largest=False,  # 最小距离
            dim=1
        )
    
    return patch_scores, locations
```

**性能优化**：
- 当`n_neighbors=1`时使用`min`代替`topk`，减少计算量
- 批量计算所有patch的距离，避免循环

### 4.3 加权异常分数计算

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第349-406行)

PatchCore使用加权机制计算图像级异常分数，考虑最近邻的局部邻域结构。

```python
def compute_anomaly_score(
    self,
    patch_scores: torch.Tensor,  # [B, H*W] - 每个patch的最近邻距离
    locations: torch.Tensor,      # [B, H*W] - 最近邻索引
    embedding: torch.Tensor,      # [B, C, H, W] - 测试特征
) -> torch.Tensor:
    """计算图像级异常分数（论文的加权评分机制）
    
    算法流程（论文Section 3.3）：
    1. 找到每个图像中距离最近邻最远的patch（m^test,*）
    2. 找到该patch的最近邻在记忆库中的位置（m^*）
    3. 找到m^*在记忆库中的支持样本（N_b(m^*)）
    4. 计算测试patch到支持样本的距离
    5. 使用softmax计算权重
    6. 加权分数 = weight * score
    
    Args:
        patch_scores: [B, num_patches] - patch级异常分数
        locations: [B, num_patches] - 最近邻在记忆库中的索引
        embedding: [B, C, H, W] - 测试图像特征
    
    Returns:
        [B] - 图像级异常分数
    """
    if self.num_neighbors == 1:
        # 当num_neighbors=1时，直接返回最大patch分数
        return patch_scores.amax(1)
    
    batch_size, num_patches = patch_scores.shape
    
    # 1. 找到每个图像中距离最远的patch（m^test,*）
    max_patches = torch.argmax(patch_scores, dim=1)  # [B]
    
    # 2. 提取这些patch的特征（m^test,*）
    max_patches_features = embedding.reshape(
        batch_size, num_patches, -1
    )[torch.arange(batch_size), max_patches]  # [B, C]
    
    # 3. 获取这些patch的最近邻距离和索引
    score = patch_scores[torch.arange(batch_size), max_patches]  # s^* [B]
    nn_index = locations[torch.arange(batch_size), max_patches]  # m^*的索引 [B]
    
    # 4. 找到最近邻在记忆库中的支持样本（N_b(m^*)）
    nn_sample = self.memory_bank[nn_index, :]  # m^* [B, C]
    
    # 查找m^*的最近邻（支持样本）
    memory_bank_effective_size = self.memory_bank.shape[0]
    _, support_samples = self.nearest_neighbors(
        nn_sample,
        n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
    )  # support_samples: [B, k]
    
    # 5. 计算测试patch到支持样本的距离
    distances = self.euclidean_dist(
        max_patches_features.unsqueeze(1),  # [B, 1, C]
        self.memory_bank[support_samples]     # [B, k, C]
    )  # [B, 1, k]
    
    # 6. 使用softmax计算权重（距离越小，权重越大）
    weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]  # [B]
    
    # 7. 加权分数
    return weights * score  # s [B]
```

**算法说明**：
- **加权机制**：考虑最近邻的局部邻域，如果测试patch与支持样本距离近，说明该区域正常，权重降低
- **softmax权重**：距离越小，softmax值越大，权重`(1 - softmax)`越小
- **最终分数**：`weight * score`，异常区域权重大，正常区域权重小

### 4.4 异常图生成

**代码来源**：`anomalib/models/image/patchcore/anomaly_map.py` (第37-111行)

```python
class AnomalyMapGenerator(nn.Module):
    """生成异常热力图"""
    
    def __init__(self, sigma: int = 4) -> None:
        super().__init__()
        # 高斯模糊核大小
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            channels=1
        )
    
    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,  # [B, 1, H, W] - patch级异常分数
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """计算像素级异常热力图
        
        Args:
            patch_scores: [B, 1, H, W] - patch级异常分数
            image_size: 目标图像尺寸，用于上采样
        
        Returns:
            [B, 1, H', W'] - 异常热力图（上采样并平滑后）
        """
        if image_size is None:
            anomaly_map = patch_scores
        else:
            # 上采样到原始图像尺寸
            anomaly_map = F.interpolate(
                patch_scores, 
                size=(image_size[0], image_size[1])
            )
        
        # 高斯模糊平滑
        return self.blur(anomaly_map)
```

### 4.5 完整推理流程

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第152-195行)

```python
def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
    """前向传播"""
    output_size = input_tensor.shape[-2:]
    
    if self.tiler:
        input_tensor = self.tiler.tile(input_tensor)
    
    with torch.no_grad():
        # 1. 特征提取
        features = self.feature_extractor(input_tensor)
        embeddings = self.generate_embedding(features)
    
    if self.tiler:
        embeddings = self.tiler.untile(embeddings)
    
    if self.training:
        # 训练时：收集特征
        self.embedding_store.append(embeddings)
        return embeddings
    
    # 推理时：异常检测
    # 2. 重塑为patch特征
    batch_size, channels, height, width = embeddings.shape
    embedding = self.reshape_embedding(embeddings)  # [B*H*W, C]
    
    # 3. 最近邻搜索
    patch_scores, locations = self.nearest_neighbors(
        embedding, 
        n_neighbors=self.num_neighbors
    )  # patch_scores: [B*H*W, k], locations: [B*H*W, k]
    
    # 4. 重塑为空间图
    patch_scores = patch_scores.reshape(batch_size, height, width)
    patch_scores = patch_scores.unsqueeze(1)  # [B, 1, H, W]
    
    # 5. 生成异常图
    anomaly_map = self.anomaly_map_generator(
        patch_scores=patch_scores,
        image_size=output_size
    )
    
    # 6. 计算图像级异常分数
    if self.num_neighbors == 1:
        pred_score = patch_scores.amax(dim=(-2, -1))
    else:
        pred_score = self.compute_anomaly_score(
            patch_scores.squeeze(1),  # [B, H, W]
            locations.reshape(batch_size, height, width),  # [B, H, W]
            embeddings
        )
    
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map.squeeze(1))
```

## 五、关键实现细节与优化

### 5.1 核心集选择优化

**稀疏随机投影降维**：

**代码来源**：`anomalib/models/components/sampling/k_center_greedy.py` (第50行)

```python
# 在KCenterGreedy初始化时
self.model = SparseRandomProjection(eps=0.9)
```

K-Center Greedy算法在计算距离前使用稀疏随机投影降维，大幅加速距离计算，同时保持距离关系的近似。

### 5.2 内存优化

1. **核心集选择**：只保留10%的patch，内存占用减少90%
2. **分批处理**：训练时逐batch收集特征，避免一次性加载所有数据
3. **及时释放**：核心集选择后立即清空`embedding_store`

### 5.3 计算优化

1. **高效距离计算**：使用矩阵运算代替循环
2. **批量最近邻搜索**：一次性计算所有patch的距离
3. **条件优化**：当`num_neighbors=1`时使用`min`代替`topk`

## 六、完整训练与推理流程

### 6.1 训练流程

**代码来源**：`anomalib/models/image/patchcore/lightning_model.py`

```python
# 1. 初始化模型
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,  # 核心集比例
    num_neighbors=9  # 最近邻数量
)

# 2. 训练阶段（收集特征）
def training_step(self, batch: Batch, *args, **kwargs) -> None:
    _ = self.model(batch.image)  # 特征收集到embedding_store
    return torch.tensor(0.0, requires_grad=True, device=self.device)

# 3. 训练结束后构建记忆库
def fit(self) -> None:
    self.model.subsample_embedding(sampling_ratio=self.coreset_sampling_ratio)
```

**流程总结**：
1. 遍历所有正常训练图像
2. 提取多尺度特征并池化
3. 收集所有特征到`embedding_store`
4. 训练结束后，使用K-Center Greedy选择核心集
5. 构建记忆库并清空`embedding_store`

### 6.2 推理流程

```python
# 1. 加载模型（包含记忆库）
model = Patchcore.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 对测试图像进行前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

**推理步骤**：
1. 提取测试图像的多尺度特征并池化
2. 对每个patch在记忆库中搜索最近邻
3. 使用最近邻距离作为patch级异常分数
4. 上采样到原始图像尺寸
5. 高斯模糊平滑
6. 计算图像级异常分数（加权或最大值）

## 七、算法优缺点分析

### 7.1 优点

1. **内存高效**：核心集选择大幅减少内存占用（通常减少90%）
2. **高召回率**：K-Center Greedy保证记忆库的覆盖性
3. **计算高效**：批量距离计算，推理速度快
4. **无需训练**：直接使用预训练特征，无需反向传播
5. **加权机制**：考虑局部邻域的加权评分，提升准确性

### 7.2 缺点

1. **核心集选择耗时**：K-Center Greedy算法需要迭代计算
2. **距离计算复杂度**：O(N×M)，N为测试patch数，M为记忆库大小
3. **固定阈值**：需要手动设置异常阈值
4. **特征依赖**：依赖预训练特征的质量

## 八、关键超参数

### 8.1 模型超参数

- **backbone**：特征提取器（`"wide_resnet50_2"` 或 `"resnet18"`）
- **layers**：提取特征的层（默认`["layer2", "layer3"]`）
- **coreset_sampling_ratio**：核心集比例（默认0.1，即10%）
- **num_neighbors**：最近邻数量（默认9，用于加权评分）
- **pre_trained**：是否使用预训练权重（默认`True`）

### 8.2 后处理超参数

- **sigma**：高斯模糊的标准差（默认4，在`AnomalyMapGenerator`中）

### 8.3 论文推荐配置

根据论文Table 1：
- **Backbone**：Wide ResNet-50-2
- **Layers**：["layer2", "layer3"]
- **Coreset Ratio**：0.1（10%）
- **Num Neighbors**：9（用于加权评分）

## 九、总结

PatchCore是PaDiM的改进版本，核心创新包括：

1. **核心集选择**：使用K-Center Greedy算法选择代表性patch，大幅减少内存占用
2. **特征池化**：使用Adaptive Average Pooling统一特征尺寸
3. **加权评分**：考虑最近邻的局部邻域，提升检测准确性
4. **高效实现**：批量距离计算，支持ONNX导出

该方法在保持高召回率的同时，大幅减少了内存占用，是工业异常检测的实用方法。
```

            pre_trained=pre_trained,
            layers=self.layers,
        ).eval()
        
        # 特征池化（3x3平均池化，stride=1，padding=1）
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        
        # 记忆库（使用register_buffer注册，支持模型保存/加载）
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.tensor] = []
        
        # 异常图生成器
        self.anomaly_map_generator = AnomalyMapGenerator()
```

**关键设计**：
- `DynamicBufferMixin` 提供动态buffer管理，支持模型序列化
- `feature_pooler` 使用3x3平均池化平滑特征，减少噪声
- `embedding_store` 临时存储训练特征，训练后通过coreset选择

#### 10.2 特征嵌入生成

**代码来源**：`anomalib/models/image/patchcore/torch_model.py` (第196-225行)

```python
def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
    """生成多尺度特征嵌入"""
    # 从第一个层开始
    embeddings = features[self.layers[0]]
    
    # 拼接其他层特征（上采样到相同尺寸）
    for layer in self.layers[1:]:
        layer_embedding = features[layer]
        layer_embedding = F.interpolate(
            layer_embedding, 
            size=embeddings.shape[-2:], 
            mode="bilinear"
        )
        embeddings = torch.cat((embeddings, layer_embedding), 1)
    
    return embeddings
```


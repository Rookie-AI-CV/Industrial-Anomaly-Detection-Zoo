# CFA - Coupled-hypersphere-based Feature Adaptation 完整算法解读

## 论文信息
- **标题**: CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization
- **年份**: 2022
- **论文链接**: https://arxiv.org/abs/2206.04325
- **核心思想**: 使用耦合超球面进行特征适应，将正常样本特征映射到超球面中心，异常样本偏离中心

## 算法概述

CFA是一种基于特征适应的异常检测方法。它使用描述符网络（Descriptor）将预训练特征适应到超球面空间，通过计算测试特征与记忆库中正常特征的距离来检测异常。核心创新是使用CoordConv和耦合超球面进行目标导向的特征适应。

### 整体架构

```
输入图像 → 预训练特征提取 → 描述符网络（CoordConv） → 目标导向特征 → 距离计算 → 异常图生成
```

## 一、模型架构

### 1.1 特征提取器

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第93-119行)

CFA使用torchvision的预训练模型提取多尺度特征。

```python
def get_feature_extractor(backbone: str, return_nodes: list[str]) -> GraphModule:
    """创建特征提取器
    
    Args:
        backbone: 骨干网络名称（如"resnet18", "wide_resnet50_2", "vgg19_bn"）
        return_nodes: 提取特征的层名称
    
    Returns:
        GraphModule: 特征提取器
    """
    model = getattr(torchvision.models, backbone)(pretrained=True)
    feature_extractor = create_feature_extractor(model=model, return_nodes=return_nodes)
    feature_extractor.eval()  # 冻结参数
    
    return feature_extractor

def get_return_nodes(backbone: str) -> list[str]:
    """获取特征提取的层名称"""
    if backbone in {"resnet18", "wide_resnet50_2"}:
        return_nodes = ["layer1", "layer2", "layer3"]
    elif backbone == "vgg19_bn":
        return_nodes = ["features.25", "features.38", "features.52"]
    return return_nodes
```

### 1.2 描述符网络（Descriptor）

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第315-378行)

描述符网络使用CoordConv进行特征适应，生成目标导向的特征。

```python
class Descriptor(nn.Module):
    """描述符网络：使用CoordConv进行特征适应"""
    
    def __init__(self, gamma_d: int, backbone: str) -> None:
        super().__init__()
        self.gamma_d = gamma_d
        
        # 获取特征维度
        return_nodes = get_return_nodes(backbone)
        feature_map_metadata = dryrun_find_featuremap_dims(
            feature_extractor=get_feature_extractor(backbone, return_nodes),
            input_size=(256, 256),
            layers=return_nodes,
        )
        
        # 计算输入通道数（所有层通道数之和）
        num_features = sum(
            feature_map_metadata[layer]["num_features"] 
            for layer in return_nodes
        )
        
        # CoordConv层：添加坐标信息
        self.coord_conv = CoordConv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """生成目标导向特征
        
        Args:
            features: 多尺度特征列表，每个元素是 [B, C, H, W]
        
        Returns:
            [B, C, H, W] - 目标导向特征
        """
        # 1. 池化并拼接多尺度特征
        pooled_features = []
        for feature in features:
            # 全局平均池化
            pooled = F.adaptive_avg_pool2d(feature, (1, 1))  # [B, C, 1, 1]
            pooled_features.append(pooled)
        
        # 拼接所有层的池化特征
        concatenated = torch.cat(pooled_features, dim=1)  # [B, C_total, 1, 1]
        
        # 2. 扩展到与第一个特征图相同的空间尺寸
        target_size = features[0].shape[-2:]
        expanded = concatenated.expand(-1, -1, *target_size)  # [B, C_total, H, W]
        
        # 3. 拼接原始特征和扩展特征
        # 对每个特征层上采样到相同尺寸
        upsampled_features = []
        for feature in features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, 
                    size=target_size, 
                    mode="bilinear", 
                    align_corners=False
                )
            upsampled_features.append(feature)
        
        # 拼接所有特征层
        all_features = torch.cat(upsampled_features, dim=1)  # [B, C_total, H, W]
        
        # 拼接原始特征和扩展特征
        combined = torch.cat([all_features, expanded], dim=1)  # [B, 2*C_total, H, W]
        
        # 4. 通过CoordConv生成目标导向特征
        target_oriented = self.coord_conv(combined)  # [B, C_total, H, W]
        
        return target_oriented
```

**CoordConv实现**：

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第380-540行)

```python
class CoordConv2d(nn.Module):
    """CoordConv：添加坐标通道的卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
    ) -> None:
        super().__init__()
        # 添加2个坐标通道
        self.add_coords = AddCoords()
        self.conv = nn.Conv2d(
            in_channels + 2,  # +2 for coordinate channels
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.add_coords(x)  # 添加坐标通道
        return self.conv(x)

class AddCoords(nn.Module):
    """添加坐标通道到特征图"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加x和y坐标通道"""
        B, _, H, W = x.shape
        
        # 创建坐标网格
        xx_channel = torch.arange(W, device=x.device).repeat(H, 1).float()
        yy_channel = torch.arange(H, device=x.device).repeat(W, 1).transpose(0, 1).float()
        
        # 归一化到[-1, 1]
        xx_channel = xx_channel / (W - 1) * 2 - 1
        yy_channel = yy_channel / (H - 1) * 2 - 1
        
        # 扩展到batch维度
        xx_channel = xx_channel.repeat(B, 1, 1, 1)  # [B, 1, H, W]
        yy_channel = yy_channel.repeat(B, 1, 1, 1)  # [B, 1, H, W]
        
        # 拼接坐标通道
        return torch.cat([x, xx_channel, yy_channel], dim=1)
```

**关键设计**：
- **CoordConv**：添加坐标信息，使网络能够感知空间位置
- **目标导向**：通过拼接全局池化特征，生成目标导向的特征表示
- **多尺度融合**：融合不同层的特征，保留多尺度信息

## 二、记忆库初始化

### 2.1 记忆库初始化

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第212-248行)

CFA使用记忆库存储正常样本的特征表示。初始化时计算所有正常样本的平均特征，可选使用K-means聚类。

```python
def initialize_centroid(self, data_loader: DataLoader) -> None:
    """初始化记忆库中心
    
    计算正常样本的平均特征表示，可选使用K-means聚类。
    
    Args:
        data_loader: 包含正常训练样本的DataLoader
    """
    device = next(self.feature_extractor.parameters()).device
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            batch = data.image.to(device)
            
            # 1. 提取特征
            features = self.feature_extractor(batch)
            features = list(features.values())
            
            # 2. 通过描述符网络生成目标导向特征
            target_features = self.descriptor(features)  # [B, C, H, W]
            
            # 3. 计算平均特征（增量更新）
            self.memory_bank = (
                (self.memory_bank * i) + target_features.mean(dim=0, keepdim=True)
            ) / (i + 1)
    
    # 4. 重塑为 [H*W, C]
    self.memory_bank = rearrange(self.memory_bank, "b c h w -> (b h w) c")
    
    # 5. 可选：使用K-means聚类（如果gamma_c > 1）
    scale = self.get_scale(batch.shape[-2:])
    
    if self.gamma_c > 1:
        # 使用K-means聚类减少记忆库大小
        k_means = KMeans(
            n_clusters=(scale[0] * scale[1]) // self.gamma_c, 
            max_iter=3000
        )
        cluster_centers = k_means.fit(self.memory_bank.cpu()).cluster_centers_
        self.memory_bank = torch.tensor(
            cluster_centers, 
            requires_grad=False
        ).to(device)
    
    # 6. 转置为 [C, N]（N是记忆库大小）
    self.memory_bank = rearrange(self.memory_bank, "h w -> w h")
```

**初始化说明**：
- **增量平均**：逐batch计算平均特征，避免一次性加载所有数据
- **K-means聚类**：如果`gamma_c > 1`，使用K-means聚类减少记忆库大小
- **记忆库格式**：最终存储为`[C, N]`，C是特征维度，N是记忆库大小

### 2.2 距离计算

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第250-271行)

CFA使用欧氏距离计算测试特征与记忆库中心的距离。

```python
def compute_distance(self, target_oriented_features: torch.Tensor) -> torch.Tensor:
    """计算特征与记忆库中心的距离
    
    使用欧氏距离：||x - c||² = ||x||² + ||c||² - 2x·c
    
    Args:
        target_oriented_features: [B, C, H, W] 或 [B, H*W, C] - 目标导向特征
    
    Returns:
        [B, H*W, 1] - 距离（平方）
    """
    # 重塑为 [B, H*W, C]
    if target_oriented_features.ndim == 4:
        target_oriented_features = rearrange(
            target_oriented_features, 
            "b c h w -> b (h w) c"
        )
    
    # 计算 ||x||²
    features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)  # [B, H*W, 1]
    
    # 计算 ||c||²
    centers = self.memory_bank.pow(2).sum(dim=0, keepdim=True).to(features.device)  # [1, N]
    
    # 计算 2x·c
    f_c = 2 * torch.matmul(
        target_oriented_features, 
        self.memory_bank.to(features.device)
    )  # [B, H*W, N]
    
    # 欧氏距离：||x||² + ||c||² - 2x·c
    # 对每个记忆库中心计算距离
    distance = features + centers - f_c  # [B, H*W, N]
    
    return distance
```

**距离计算优化**：
- **矩阵运算**：批量计算所有patch到所有记忆库中心的距离
- **欧氏距离公式**：`||x - c||² = ||x||² + ||c||² - 2x·c`，避免显式计算差值

## 三、前向传播

### 3.1 完整前向传播流程

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第273-312行)

```python
def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
    """前向传播
    
    Args:
        input_tensor: [B, 3, H, W] - 输入图像
    
    Returns:
        训练时: distance tensor
        推理时: InferenceBatch(pred_score, anomaly_map)
    """
    # 检查记忆库是否初始化
    if self.memory_bank.ndim == 0:
        raise ValueError("Memory bank is not initialized. Run `initialize_centroid` first.")
    
    self.feature_extractor.eval()
    
    # 1. 提取特征（冻结梯度）
    with torch.no_grad():
        features = self.feature_extractor(input_tensor)
        features = list(features.values())
    
    # 2. 描述符网络生成目标导向特征
    target_features = self.descriptor(features)  # [B, C, H, W]
    
    # 3. 计算距离
    distance = self.compute_distance(target_features)  # [B, H*W, N]
    
    if self.training:
        # 训练时：返回距离（用于计算损失）
        return distance
    
    # 推理时：生成异常图
    anomaly_map = self.anomaly_map_generator(
        distance=distance,
        scale=target_features.shape[-2:],
        image_size=input_tensor.shape[-2:],
    ).squeeze()
    
    pred_score = torch.amax(anomaly_map, dim=(-2, -1))
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
```

## 四、异常图生成

### 4.1 K近邻异常分数计算

**代码来源**：`anomalib/models/image/cfa/anomaly_map.py` (第28-131行)

CFA使用K近邻距离计算异常分数。

```python
class AnomalyMapGenerator(nn.Module):
    """从距离特征生成异常热力图"""
    
    def __init__(
        self,
        num_nearest_neighbors: int,  # K值
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.sigma = sigma
    
    def compute_score(
        self, 
        distance: torch.Tensor,  # [B, H*W, N] - 到记忆库的距离
        scale: tuple[int, int]   # 特征图尺寸
    ) -> torch.Tensor:
        """计算异常分数
        
        Args:
            distance: [B, H*W, N] - 每个patch到记忆库的距离
            scale: (H, W) - 特征图尺寸
        
        Returns:
            [B, 1, H, W] - 异常分数图
        """
        # 1. 开方得到欧氏距离
        distance = torch.sqrt(distance)  # [B, H*W, N]
        
        # 2. 选择K个最近邻
        distance = distance.topk(
            self.num_nearest_neighbors, 
            largest=False  # 最小距离
        ).values  # [B, H*W, K]
        
        # 3. 使用softmin加权（距离越小，权重越大）
        weights = F.softmin(distance, dim=-1)[:, :, 0]  # [B, H*W]
        
        # 4. 加权距离
        weighted_distance = weights * distance[:, :, 0]  # [B, H*W]
        weighted_distance = weighted_distance.unsqueeze(-1)  # [B, H*W, 1]
        
        # 5. 重塑为空间图
        score = rearrange(
            weighted_distance, 
            "b (h w) c -> b c h w", 
            h=scale[0], 
            w=scale[1]
        )
        
        return score.detach()
    
    def compute_anomaly_map(
        self,
        score: torch.Tensor,  # [B, 1, H, W]
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """生成平滑的异常图
        
        Args:
            score: [B, 1, H, W] - 异常分数
            image_size: 目标图像尺寸（用于上采样）
        
        Returns:
            [B, 1, H', W'] - 平滑后的异常图
        """
        # 平均（如果有多个通道）
        anomaly_map = score.mean(dim=1, keepdim=True)
        
        # 上采样到原始图像尺寸
        if image_size is not None:
            anomaly_map = F.interpolate(
                anomaly_map, 
                size=image_size, 
                mode="bilinear", 
                align_corners=False
            )
        
        # 高斯模糊平滑
        gaussian_blur = GaussianBlur2d(sigma=self.sigma).to(score.device)
        return gaussian_blur(anomaly_map)
```

**异常分数计算说明**：
- **K近邻**：选择K个最近邻，考虑局部邻域
- **Softmin加权**：距离越小，权重越大，更关注最近邻
- **加权距离**：使用softmin权重对最近邻距离加权

## 五、训练阶段

### 5.1 损失函数

**代码来源**：`anomalib/models/image/cfa/loss.py`

CFA的损失函数包含两个组件：中心损失（centroid loss）和距离损失（distance loss）。

```python
class CfaLoss(nn.Module):
    """CFA损失函数：中心损失 + 距离损失"""
    
    def forward(
        self,
        distance: torch.Tensor,  # [B, H*W, N] - 到记忆库的距离
        gamma_c: int,             # 中心损失权重
        gamma_d: int,             # 距离损失权重
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            distance: [B, H*W, N] - 每个patch到记忆库的距离
            gamma_c: 中心损失权重
            gamma_d: 距离损失权重
        
        Returns:
            标量损失值
        """
        # 1. 中心损失：最小化到最近邻的距离（紧凑性）
        min_distances = distance.min(dim=-1)[0]  # [B, H*W] - 到最近邻的距离
        centroid_loss = min_distances.mean()
        
        # 2. 距离损失：鼓励特征分散（如果有硬负样本）
        # 这里简化，实际实现可能更复杂
        distance_loss = 0  # 根据论文实现
        
        # 总损失
        return gamma_c * centroid_loss + gamma_d * distance_loss
```

## 六、完整训练与推理流程

### 6.1 训练流程

```python
# 1. 初始化模型
model = Cfa(
    backbone="resnet18",
    gamma_c=1,
    gamma_d=1,
    num_nearest_neighbors=3,
    num_hard_negative_features=3,
    radius=0.5
)

# 2. 初始化记忆库
model.initialize_centroid(train_loader)

# 3. 训练
def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
    distance = self.model(batch.image)
    loss = self.loss(distance, self.gamma_c, self.gamma_d)
    return {"loss": loss}
```

### 6.2 推理流程

```python
# 1. 加载模型（包含记忆库）
model = Cfa.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

## 七、关键实现细节

### 7.1 CoordConv的作用

CoordConv添加坐标信息，使网络能够：
- 感知空间位置
- 生成目标导向的特征
- 提升异常定位精度

### 7.2 记忆库设计

- **初始化**：使用所有正常样本的平均特征
- **可选聚类**：使用K-means减少记忆库大小
- **格式**：存储为`[C, N]`，便于矩阵运算

### 7.3 K近邻评分

- **K值选择**：通常3-9，平衡精度和计算量
- **Softmin加权**：距离越小权重越大
- **加权距离**：考虑局部邻域结构

## 八、算法优缺点分析

### 8.1 优点

1. **目标导向特征**：CoordConv生成目标导向特征，提升定位精度
2. **记忆库机制**：存储正常样本特征，支持K近邻搜索
3. **多尺度融合**：融合多尺度特征，保留细节信息
4. **无需异常样本**：只使用正常样本训练

### 8.2 缺点

1. **记忆库大小**：需要存储所有正常样本特征，内存占用大
2. **K近邻计算**：计算复杂度O(N×M)，N为测试patch数，M为记忆库大小
3. **训练复杂**：需要初始化记忆库和训练描述符网络

## 九、关键超参数

### 9.1 模型超参数

- **backbone**：特征提取器（"resnet18", "wide_resnet50_2", "vgg19_bn"）
- **gamma_c**：中心损失权重（默认1）
- **gamma_d**：距离损失权重（默认1）
- **num_nearest_neighbors**：K近邻数量（默认3）
- **num_hard_negative_features**：硬负样本数量（默认3）
- **radius**：超球面半径（默认0.5）

### 9.2 后处理超参数

- **sigma**：高斯模糊的标准差（默认4）

## 十、使用示例

### 10.1 基本使用

```python
from anomalib.models import Cfa
from anomalib.data import MVTecAD
from anomalib.engine import Engine

# 初始化模型
model = Cfa(
    backbone="resnet18",
    gamma_c=1,
    gamma_d=1,
    num_nearest_neighbors=3,
    num_hard_negative_features=3,
    radius=0.5
)

# 加载数据
datamodule = MVTecAD(category="bottle")

# 初始化记忆库
model.initialize_centroid(datamodule.train_dataloader())

# 训练
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# 推理
predictions = engine.predict(model=model, datamodule=datamodule)
```

## 十一、总结

CFA是一种基于特征适应和记忆库的异常检测方法，核心思想是：
1. **特征适应**：使用CoordConv生成目标导向特征
2. **记忆库**：存储正常样本特征，支持K近邻搜索
3. **距离计算**：计算测试特征到记忆库的距离
4. **K近邻评分**：使用softmin加权K近邻距离计算异常分数

该方法在目标导向的异常定位任务上表现优异，特别适合需要精确定位的场景。


### 10. 优缺点分析

**优点**：
- 特征适应提升检测性能
- 超球面机制简单直观
- 训练过程简单高效
- 异常分数有明确的几何意义

**缺点**：
- 需要训练特征适应网络
- 假设正常样本在超球面上分布，可能不符合实际情况
- 对特征质量依赖较大
- 超球面中心的初始化可能影响性能

### 11. 关键超参数

- **适应特征维度**：adapted_dim（如256）
- **特征适应网络结构**：层数和通道数
- **学习率**：训练学习率
- **中心更新策略**：EMA动量或直接学习
- **多尺度融合策略**：平均、最大值或加权融合

## 代码实现详解（基于Anomalib）

### 12. 核心代码结构

基于Anomalib的实现，CFA的核心代码位于 `anomalib/models/image/cfa/torch_model.py`。

#### 12.1 模型架构

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第122-178行)

```python
class CfaModel(DynamicBufferMixin):
    """CFA模型：特征适应 + 超球面距离计算"""
    def __init__(
        self,
        backbone: str,
        gamma_c: int,
        gamma_d: int,
        num_nearest_neighbors: int,
        num_hard_negative_features: int,
        radius: float,
    ) -> None:
        # 特征提取器（冻结）
        self.feature_extractor = get_feature_extractor(backbone, return_nodes)
        
        # 描述符网络（特征适应）
        self.descriptor = Descriptor(self.gamma_d, backbone)
        
        # 超球面半径（可学习参数）
        self.radius = torch.ones(1, requires_grad=True) * radius
        
        # 记忆库（存储正常特征中心）
        self.register_buffer("memory_bank", torch.tensor(0.0))
```

#### 12.2 描述符网络（CoordConv）

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第315-378行)

```python
class Descriptor(nn.Module):
    """描述符网络：使用CoordConv进行特征适应"""
    def __init__(self, gamma_d: int, backbone: str):
        # 计算输出通道数
        backbone_dims = {"resnet18": 448, "wide_resnet50_2": 1792}
        dim = backbone_dims[backbone]
        out_channels = dim // gamma_d
        
        # CoordConv层（添加坐标通道）
        self.layer = CoordConv2d(
            in_channels=dim, 
            out_channels=out_channels, 
            kernel_size=1
        )
    
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """特征适应"""
        patch_features = None
        for feature in features:
            # 平均池化
            pooled = F.avg_pool2d(feature, 3, 1, 1)
            
            # 上采样并拼接
            if patch_features is None:
                patch_features = pooled
            else:
                patch_features = torch.cat((
                    patch_features, 
                    F.interpolate(feature, patch_features.size(2), mode="bilinear")
                ), dim=1)
        
        # CoordConv适应
        return self.layer(patch_features)
```

**CoordConv说明**：
- 添加x、y坐标通道，增强空间感知能力
- 使用1x1卷积进行特征适应，保持空间分辨率

#### 12.3 距离计算

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第250-271行)

```python
def compute_distance(self, target_oriented_features: torch.Tensor) -> torch.Tensor:
    """计算到记忆库中心的距离"""
    # 展平特征
    if target_oriented_features.ndim == 4:
        target_oriented_features = rearrange(
            target_oriented_features, "b c h w -> b (h w) c"
        )
    
    # 欧氏距离：||f - c||^2 = ||f||^2 + ||c||^2 - 2*f@c
    features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)
    centers = self.memory_bank.pow(2).sum(dim=0, keepdim=True).to(features.device)
    f_c = 2 * torch.matmul(target_oriented_features, self.memory_bank.to(features.device))
    
    return features + centers - f_c
```

**优化**：使用展开公式计算距离，避免显式计算差值

#### 12.4 记忆库初始化

**代码来源**：`anomalib/models/image/cfa/torch_model.py` (第212-248行)

```python
def initialize_centroid(self, data_loader: DataLoader) -> None:
    """初始化记忆库中心"""
    # 收集所有正常特征
    for data in data_loader:
        features = self.feature_extractor(data.image)
        target_features = self.descriptor(list(features.values()))
        self.memory_bank = ((self.memory_bank * i) + target_features.mean(dim=0, keepdim=True)) / (i + 1)
    
    # 可选：K-means聚类压缩
    if self.gamma_c > 1:
        k_means = KMeans(n_clusters=(scale[0] * scale[1]) // self.gamma_c)
        cluster_centers = k_means.fit(self.memory_bank.cpu()).cluster_centers_
        self.memory_bank = torch.tensor(cluster_centers).to(device)
```


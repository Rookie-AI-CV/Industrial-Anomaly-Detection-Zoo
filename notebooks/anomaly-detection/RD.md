# RD - Reverse Distillation from One-Class Embedding 完整算法解读

## 论文信息
- **标题**: Anomaly Detection via Reverse Distillation from One-Class Embedding
- **年份**: 2022
- **论文链接**: https://arxiv.org/abs/2201.10703
- **核心思想**: 采用反向蒸馏架构，编码器提取多尺度特征，通过瓶颈层压缩后，解码器重建特征，使用重建误差检测异常

## 算法概述

RD（Reverse Distillation）是一种基于知识蒸馏的异常检测方法。与传统蒸馏不同，RD使用反向蒸馏：编码器（教师）提取特征，解码器（学生）重建特征。正常样本能很好地重建，异常样本重建困难，产生较大误差。

### 整体架构

```
输入图像 → 编码器（预训练） → 多尺度特征 → 瓶颈层 → 解码器 → 重建特征 → 异常图生成
```

## 一、模型架构

### 1.1 编码器-瓶颈-解码器结构

**代码来源**：`anomalib/models/image/reverse_distillation/torch_model.py` (第51-112行)

RD使用编码器-瓶颈-解码器架构，其中编码器是预训练的，瓶颈层压缩特征，解码器重建特征。

```python
class ReverseDistillationModel(nn.Module):
    """反向蒸馏模型
    
    Args:
        backbone: 骨干网络（如"wide_resnet50_2"）
        input_size: 输入图像尺寸 (H, W)
        layers: 提取特征的层（如["layer1", "layer2", "layer3"]）
        anomaly_map_mode: 异常图生成模式（"multiply"或"add"）
        pre_trained: 是否使用预训练权重
    """
    
    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: Sequence[str],
        anomaly_map_mode: AnomalyMapGenerationMode,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None
        
        # 编码器：使用TimmFeatureExtractor提取多尺度特征
        self.encoder = TimmFeatureExtractor(
            backbone=backbone, 
            pre_trained=pre_trained, 
            layers=layers
        )
        
        # 瓶颈层：压缩特征
        self.bottleneck = get_bottleneck_layer(backbone)
        
        # 解码器：重建特征
        self.decoder = get_decoder(backbone)
        
        # 异常图生成器
        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=input_size, 
            mode=anomaly_map_mode
        )
```

**关键组件**：
- **编码器**：预训练的CNN（如Wide ResNet-50-2），提取多尺度特征
- **瓶颈层**：压缩特征维度，学习紧凑表示
- **解码器**：从压缩特征重建原始特征空间

### 1.2 前向传播

**代码来源**：`anomalib/models/image/reverse_distillation/torch_model.py` (第114-168行)

```python
def forward(
    self, 
    images: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
    """前向传播
    
    Args:
        images: [B, 3, H, W] - 输入图像
    
    Returns:
        训练时: (encoder_features, decoder_features)
        推理时: InferenceBatch(pred_score, anomaly_map)
    """
    self.encoder.eval()  # 编码器冻结，不计算梯度
    
    # 支持大图像分块处理
    if self.tiler:
        images = self.tiler.tile(images)
    
    # 1. 编码器提取多尺度特征
    encoder_features = self.encoder(images)  # dict
    encoder_features = list(encoder_features.values())  # list of [B, C, H, W]
    
    # 2. 瓶颈层压缩特征
    compressed = self.bottleneck(encoder_features)
    
    # 3. 解码器重建特征
    decoder_features = self.decoder(compressed)
    
    if self.tiler:
        # 合并分块
        for i, features in enumerate(encoder_features):
            encoder_features[i] = self.tiler.untile(features)
        for i, features in enumerate(decoder_features):
            decoder_features[i] = self.tiler.untile(features)
    
    if self.training:
        # 训练时：返回编码器和解码器特征（用于计算损失）
        return encoder_features, decoder_features
    
    # 推理时：生成异常图
    anomaly_map = self.anomaly_map_generator(encoder_features, decoder_features)
    pred_score = torch.amax(anomaly_map, dim=(-2, -1))
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
```

**关键设计**：
- **编码器冻结**：编码器使用`eval()`模式，不计算梯度，只提取特征
- **多尺度特征**：从多个层提取特征，保留不同语义信息
- **特征重建**：解码器重建编码器特征，正常样本重建误差小

## 二、损失函数

### 2.1 余弦相似度损失

**代码来源**：`anomalib/models/image/reverse_distillation/loss.py` (第35-90行)

RD使用余弦相似度损失，衡量编码器和解码器特征之间的差异。

```python
class ReverseDistillationLoss(nn.Module):
    """反向蒸馏损失函数：基于余弦相似度"""
    
    @staticmethod
    def forward(
        encoder_features: list[torch.Tensor],  # 编码器特征列表
        decoder_features: list[torch.Tensor]    # 解码器特征列表
    ) -> torch.Tensor:
        """计算余弦相似度损失
        
        Args:
            encoder_features: 编码器特征列表，每个元素是 [B, C, H, W]
            decoder_features: 解码器特征列表，必须与编码器特征匹配
        
        Returns:
            标量损失值：mean(1 - cosine_similarity)
        """
        cos_loss = torch.nn.CosineSimilarity()
        loss_sum = 0
        
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            # 展平空间维度：[B, C, H, W] -> [B, C*H*W]
            encoder_flat = encoder_feature.view(encoder_feature.shape[0], -1)
            decoder_flat = decoder_feature.view(decoder_feature.shape[0], -1)
            
            # 计算余弦相似度，然后计算不相似度
            loss_sum += torch.mean(
                1 - cos_loss(encoder_flat, decoder_flat)
            )
        
        return loss_sum
```

**损失计算说明**：
- **余弦相似度**：衡量编码器和解码器特征的方向相似性
- **不相似度**：`1 - cosine_similarity`，值越大表示差异越大
- **多尺度融合**：对所有特征层的损失求和

### 2.2 训练流程

**代码来源**：`anomalib/models/image/reverse_distillation/lightning_model.py`

```python
def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
    """训练步骤"""
    # 前向传播（返回编码器和解码器特征）
    encoder_features, decoder_features = self.model(batch.image)
    
    # 计算损失
    loss = self.loss(encoder_features, decoder_features)
    
    return {"loss": loss}
```

**训练特点**：
- 编码器冻结（`encoder.eval()`），不计算梯度
- 只训练瓶颈层和解码器
- 使用余弦相似度损失，鼓励解码器重建编码器特征

## 三、异常图生成

### 3.1 异常图生成器

**代码来源**：`anomalib/models/image/reverse_distillation/anomaly_map.py` (第51-117行)

RD通过计算编码器和解码器特征的余弦不相似度生成异常图。

```python
class AnomalyMapGenerator(nn.Module):
    """从编码器和解码器特征生成异常热力图"""
    
    def __init__(
        self,
        image_size: ListConfig | tuple,
        sigma: int = 4,
        mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.MULTIPLY,
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.mode = mode  # "multiply" 或 "add"
    
    def forward(
        self, 
        student_features: list[torch.Tensor],  # 编码器特征
        teacher_features: list[torch.Tensor]  # 解码器特征
    ) -> torch.Tensor:
        """计算异常图
        
        Args:
            student_features: 编码器特征列表
            teacher_features: 解码器特征列表
        
        Returns:
            [B, 1, H, W] - 异常热力图
        """
        # 初始化异常图
        if self.mode == AnomalyMapGenerationMode.MULTIPLY:
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )
        else:  # ADD
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )
        
        # 对每个特征层计算距离图
        for student_feature, teacher_feature in zip(student_features, teacher_features, strict=True):
            # 计算余弦不相似度：1 - cosine_similarity
            distance_map = 1 - F.cosine_similarity(
                student_feature, 
                teacher_feature,
                dim=1  # 在通道维度计算
            )  # [B, H, W]
            
            distance_map = torch.unsqueeze(distance_map, dim=1)  # [B, 1, H, W]
            
            # 上采样到原始图像尺寸
            distance_map = F.interpolate(
                distance_map, 
                size=self.image_size, 
                mode="bilinear", 
                align_corners=True
            )
            
            # 融合多尺度距离图
            if self.mode == AnomalyMapGenerationMode.MULTIPLY:
                anomaly_map *= distance_map  # 相乘：所有尺度都异常才异常
            else:  # ADD
                anomaly_map += distance_map  # 相加：任一尺度异常就异常
        
        # 高斯模糊平滑
        gaussian_blur = GaussianBlur2d(
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma=(self.sigma, self.sigma),
        ).to(student_features[0].device)
        
        return gaussian_blur(anomaly_map)
```

**异常图生成说明**：
- **余弦不相似度**：`1 - cosine_similarity`，值越大表示差异越大
- **多尺度融合**：使用"multiply"（相乘）或"add"（相加）融合不同尺度的距离图
- **高斯模糊**：平滑异常图，减少噪声

### 3.2 融合模式

**AnomalyMapGenerationMode**：
- **MULTIPLY（相乘）**：所有尺度都异常才异常，更严格
- **ADD（相加）**：任一尺度异常就异常，更敏感

## 四、完整训练与推理流程

### 4.1 训练流程

```python
# 1. 初始化模型
model = ReverseDistillation(
    backbone="wide_resnet50_2",
    input_size=(256, 256),
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode="multiply"
)

# 2. 训练（只训练瓶颈层和解码器）
def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
    encoder_features, decoder_features = self.model(batch.image)
    loss = self.loss(encoder_features, decoder_features)
    return {"loss": loss}
```

**训练步骤**：
1. 编码器提取多尺度特征（冻结）
2. 瓶颈层压缩特征
3. 解码器重建特征
4. 计算编码器和解码器特征的余弦相似度损失
5. 反向传播更新瓶颈层和解码器参数

### 4.2 推理流程

```python
# 1. 加载模型
model = ReverseDistillation.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数（最大值）
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

**推理步骤**：
1. 编码器提取多尺度特征
2. 瓶颈层压缩特征
3. 解码器重建特征
4. 计算编码器和解码器特征的余弦不相似度
5. 融合多尺度距离图
6. 高斯模糊平滑
7. 返回异常分数和异常图

## 五、关键实现细节

### 5.1 编码器冻结

编码器使用`eval()`模式，不计算梯度：
```python
self.encoder.eval()  # 在forward中设置
```

这确保编码器参数不更新，只作为特征提取器使用。

### 5.2 瓶颈层设计

瓶颈层压缩多尺度特征到紧凑表示：
- 减少特征维度
- 学习正常模式的核心表示
- 提升解码器重建的难度（正常样本能重建，异常样本不能）

### 5.3 解码器设计

解码器从压缩特征重建原始特征空间：
- 使用转置卷积上采样
- 逐步恢复到原始特征尺寸
- 正常样本重建误差小，异常样本重建误差大

## 六、算法优缺点分析

### 6.1 优点

1. **简单有效**：编码器-瓶颈-解码器架构，实现简单
2. **无需异常样本**：只使用正常样本训练
3. **多尺度特征**：利用多尺度信息，提升检测精度
4. **余弦相似度**：对特征幅度不敏感，更关注方向

### 6.2 缺点

1. **编码器依赖**：依赖预训练编码器的质量
2. **训练复杂**：需要训练瓶颈层和解码器
3. **计算量大**：编码器和解码器的前向传播
4. **特征对齐**：需要确保编码器和解码器特征对齐

## 七、关键超参数

### 7.1 模型超参数

- **backbone**：编码器骨干网络（如"wide_resnet50_2"）
- **layers**：提取特征的层（如["layer1", "layer2", "layer3"]）
- **anomaly_map_mode**：异常图融合模式（"multiply"或"add"）
- **pre_trained**：是否使用预训练权重（默认True）

### 7.2 后处理超参数

- **sigma**：高斯模糊的标准差（默认4）

## 八、使用示例

### 8.1 基本使用

```python
from anomalib.models import ReverseDistillation
from anomalib.data import MVTecAD
from anomalib.engine import Engine

# 初始化模型
model = ReverseDistillation(
    backbone="wide_resnet50_2",
    input_size=(256, 256),
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode="multiply"
)

# 加载数据
datamodule = MVTecAD(category="bottle")

# 训练
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# 推理
predictions = engine.predict(model=model, datamodule=datamodule)
```

## 九、总结

RD是一种基于反向蒸馏的异常检测方法，核心思想是：
1. **编码器提取特征**：使用预训练CNN提取多尺度特征
2. **瓶颈层压缩**：学习紧凑的正常模式表示
3. **解码器重建**：从压缩特征重建原始特征
4. **余弦相似度**：通过编码器和解码器特征的相似度检测异常

该方法实现简单，在多个数据集上表现优异，是异常检测的经典方法之一。

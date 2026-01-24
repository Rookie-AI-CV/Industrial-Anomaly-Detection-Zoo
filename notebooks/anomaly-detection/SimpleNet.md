# SimpleNet - A Simple Network for Image Anomaly Detection and Localization 完整算法解读

## 论文信息
- **标题**: SimpleNet: A Simple Network for Image Anomaly Detection and Localization
- **年份**: 2023
- **论文链接**: https://arxiv.org/abs/2303.15140
- **核心思想**: 使用简单的特征提取器提取多尺度特征，通过浅层判别器网络区分正常和异常特征

## 注意

**SimpleNet在Anomalib中暂无官方实现**。Anomalib中实现了**SuperSimpleNet**（`anomalib/models/image/supersimplenet`），它是SimpleNet的扩展版本，在ICPR 2024和JIMS 2025发表。SuperSimpleNet在SimpleNet的基础上增加了特征级异常生成模块和分割-检测模块。

本文档中的代码示例为伪代码，用于说明SimpleNet的算法原理。如需查看实际实现，请参考SuperSimpleNet的实现。

## 整体架构

```
输入图像 → 特征提取器 → 特征适配器 → 判别器 → 异常分数
         ↓
    异常特征生成器（训练时）
```

## 详细实现过程

### 1. 特征提取阶段

#### 1.1 预训练特征提取器
- 使用预训练的Wide ResNet-50或EfficientNet
- 从多个中间层提取特征
- 保持特征图的空间结构

```python
# 伪代码
class FeatureExtractor(nn.Module):
    """
    特征提取器
    """
    def __init__(self, backbone='wide_resnet50'):
        super().__init__()
        # 加载预训练模型
        if backbone == 'wide_resnet50':
            model = wide_resnet50_2(pretrained=True)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
        else:
            # EfficientNet等
            pass
    
    def forward(self, x):
        """
        x: [B, 3, H, W]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f1 = self.layer1(x)  # [B, 256, H/4, W/4]
        f2 = self.layer2(f1)  # [B, 512, H/8, W/8]
        f3 = self.layer3(f2)  # [B, 1024, H/16, W/16]
        
        return [f1, f2, f3]
```

### 2. 特征适配器

#### 2.1 特征适配模块
- 将不同层的特征适配到统一空间
- 使用轻量级的卷积层
- 归一化特征

```python
# 伪代码
class FeatureAdapter(nn.Module):
    """
    特征适配器
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            for in_channels in in_channels_list
        ])
    
    def forward(self, features):
        """
        features: list of [B, C, H, W]
        """
        adapted_features = []
        for feat, adapter in zip(features, self.adapters):
            adapted = adapter(feat)
            adapted_features.append(adapted)
        
        # 上采样到相同尺寸
        max_h = max([f.shape[2] for f in adapted_features])
        max_w = max([f.shape[3] for f in adapted_features])
        
        upsampled = []
        for feat in adapted_features:
            if feat.shape[2:] != (max_h, max_w):
                feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear')
            upsampled.append(feat)
        
        # 拼接
        combined = torch.cat(upsampled, dim=1)  # [B, out_channels*num_scales, H, W]
        
        return combined
```

### 3. 判别器网络

#### 3.1 浅层判别器
- 使用简单的卷积网络
- 区分正常特征和异常特征
- 输出异常概率

```python
# 伪代码
class Discriminator(nn.Module):
    """
    判别器：区分正常和异常特征
    """
    def __init__(self, in_channels):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # 输出异常概率
        )
    
    def forward(self, features):
        """
        features: [B, C, H, W]
        """
        anomaly_prob = self.discriminator(features)  # [B, 1, H, W]
        return anomaly_prob
```

### 4. 异常特征生成器

#### 4.1 特征空间异常生成
- 在特征空间生成异常特征
- 通过扰动正常特征生成
- 用于训练判别器

```python
# 伪代码
class AnomalyFeatureGenerator(nn.Module):
    """
    异常特征生成器
    """
    def __init__(self, feature_dim):
        super().__init__()
        # 使用简单的MLP生成扰动
        self.generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, normal_features):
        """
        normal_features: [B, C, H, W]
        生成异常特征
        """
        B, C, H, W = normal_features.shape
        
        # 展平
        feat_flat = normal_features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # 生成扰动
        perturbation = self.generator(feat_flat)
        
        # 添加扰动
        anomalous_feat = feat_flat + perturbation
        
        # 重塑
        anomalous_feat = anomalous_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return anomalous_feat
```

#### 4.2 生成策略
- **随机扰动**：在特征空间随机扰动
- **对抗扰动**：使用对抗样本生成
- **特征混合**：混合不同图像的特征

```python
# 伪代码
def generate_anomalous_features(normal_features, method='random'):
    """
    生成异常特征
    """
    if method == 'random':
        # 随机扰动
        noise = torch.randn_like(normal_features) * 0.1
        anomalous = normal_features + noise
    
    elif method == 'adversarial':
        # 对抗扰动
        normal_features.requires_grad = True
        # 计算梯度并添加扰动
        # ...
    
    elif method == 'mixup':
        # 特征混合
        # 随机选择两个特征进行混合
        idx = torch.randperm(normal_features.shape[0])
        mixed = 0.5 * normal_features + 0.5 * normal_features[idx]
        anomalous = mixed
    
    return anomalous
```

### 5. 训练阶段

#### 5.1 损失函数设计
- **判别器损失**：区分正常和异常特征
- **特征适配损失**：保持特征的有用性
- **总损失**：L = L_discriminator + λ * L_adapter

```python
# 伪代码
def compute_loss(normal_features, anomalous_features, discriminator, adapter, lambda_adapter=0.1):
    """
    计算损失
    """
    # 适配特征
    adapted_normal = adapter(normal_features)
    adapted_anomalous = adapter(anomalous_features)
    
    # 判别器预测
    pred_normal = discriminator(adapted_normal)
    pred_anomalous = discriminator(adapted_anomalous)
    
    # 判别器损失：正常应该预测为0，异常应该预测为1
    loss_disc = F.binary_cross_entropy(
        pred_normal,
        torch.zeros_like(pred_normal)
    ) + F.binary_cross_entropy(
        pred_anomalous,
        torch.ones_like(pred_anomalous)
    )
    
    # 特征适配损失：保持特征的判别性
    # 可以使用对比损失或其他损失
    loss_adapter = 0  # 可选
    
    total_loss = loss_disc + lambda_adapter * loss_adapter
    return total_loss, loss_disc, loss_adapter
```

#### 5.2 训练流程
- 使用正常样本提取特征
- 生成异常特征
- 训练判别器区分正常和异常

```python
# 伪代码
def train_epoch(feature_extractor, adapter, discriminator, generator, train_loader, optimizer):
    feature_extractor.eval()  # 冻结特征提取器
    adapter.train()
    discriminator.train()
    generator.train()
    
    for normal_images in train_loader:
        # 提取正常特征
        with torch.no_grad():
            normal_features = feature_extractor(normal_images)
        
        # 适配特征
        adapted_normal = adapter(normal_features)
        
        # 生成异常特征
        anomalous_features = generator(adapted_normal)
        # 或使用其他生成方法
        # anomalous_features = generate_anomalous_features(adapted_normal)
        
        # 计算损失
        loss, loss_disc, loss_adapter = compute_loss(
            adapted_normal,
            anomalous_features,
            discriminator,
            adapter
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 6. 测试阶段

#### 6.1 异常检测流程
- 对测试图像提取特征
- 通过适配器和判别器
- 判别器输出异常概率

```python
# 伪代码
def detect_anomaly(feature_extractor, adapter, discriminator, test_image):
    """
    检测异常
    """
    feature_extractor.eval()
    adapter.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # 提取特征
        features = feature_extractor(test_image)
        
        # 适配特征
        adapted_features = adapter(features)
        
        # 判别器预测
        anomaly_prob = discriminator(adapted_features)  # [B, 1, H, W]
        
        # 异常分数图
        anomaly_map = anomaly_prob.squeeze(1)  # [B, H, W]
        
        # 图像级异常分数
        image_score = anomaly_map.max().item()  # 或使用 mean()
        
        return image_score, anomaly_map
```

#### 6.2 异常定位
- 判别器输出直接提供像素级异常概率
- 可以应用阈值进行二值化
- 使用后处理优化结果

### 7. 关键实现细节

#### 7.1 特征提取器冻结
- 特征提取器使用预训练权重，不进行微调
- 只训练适配器和判别器
- 减少训练参数，提升效率

#### 7.2 异常特征生成策略
- 需要生成多样化的异常特征
- 但不能偏离正常特征太远
- 平衡生成质量和训练效果

#### 7.3 判别器设计
- 判别器应该足够简单，避免过拟合
- 但要有足够的容量区分正常和异常
- 使用批归一化和正则化

### 8. 训练流程总结

```
1. 加载预训练特征提取器
2. 初始化特征适配器和判别器
3. 初始化异常特征生成器（可选）
4. 冻结特征提取器
5. 对每个训练epoch：
   a. 对正常训练图像：
      - 提取特征
      - 适配特征
      - 生成异常特征
      - 训练判别器区分正常和异常
      - 反向传播更新参数
6. 保存训练好的模型
```

### 9. 测试流程总结

```
1. 加载特征提取器、适配器和判别器
2. 对测试图像：
   - 提取特征
   - 适配特征
   - 判别器预测异常概率
   - 生成异常分数图
3. 上采样异常图到原始尺寸
4. 应用阈值进行异常判断
```

### 10. 优缺点分析

**优点**：
- 网络结构简单，易于实现
- 训练效率高
- 不需要真实异常样本
- 判别器提供直观的异常概率

**缺点**：
- 需要训练适配器和判别器
- 异常特征生成策略需要设计
- 对生成质量依赖较大
- 可能过拟合生成的异常模式

### 11. 关键超参数

- **特征适配器结构**：层数和通道数
- **判别器结构**：层数和通道数
- **异常生成方法**：随机、对抗、混合等
- **学习率**：训练学习率
- **批大小**：训练批大小

## 代码实现详解（基于Anomalib）

### 12. 核心代码结构

基于Anomalib的实现，SimpleNet（SuperSimpleNet）的核心代码位于 `anomalib/models/image/supersimplenet/torch_model.py`。

#### 12.1 模型架构

**代码来源**：`anomalib/models/image/supersimplenet/torch_model.py` (第28-58行)

```python
class SupersimplenetModel(nn.Module):
    """SuperSimpleNet模型"""
    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone: str = "wide_resnet50_2.tv_in1k",
        layers: list[str] = ["layer2", "layer3"],
        stop_grad: bool = True,
        adapt_cls_features: bool = False,
    ) -> None:
        # 特征提取器（上采样）
        self.feature_extractor = UpscalingFeatureExtractor(backbone=backbone, layers=layers)
        
        # 特征适配器
        channels = self.feature_extractor.get_channels_dim()
        self.adaptor = FeatureAdapter(channels)
        
        # 分割-检测模块（判别器）
        self.segdec = SegmentationDetectionModule(channel_dim=channels, stop_grad=stop_grad)
        
        # 异常生成器（Perlin噪声）
        self.anomaly_generator = AnomalyGenerator(
            noise_mean=0, 
            noise_std=0.015, 
            threshold=perlin_threshold
        )
```

#### 12.2 前向传播

**代码来源**：`anomalib/models/image/supersimplenet/torch_model.py` (第60-130行)

```python
def forward(
    self,
    images: torch.Tensor,
    masks: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> InferenceBatch | tuple:
    """前向传播"""
    features = self.feature_extractor(images)
    adapted = self.adaptor(features)
    
    if self.training:
        # 训练时：生成异常并训练判别器
        if masks is None:
            masks = torch.zeros((b, 1, h, w), device=features.device)
        
        # 生成异常特征
        _, noised_adapt, masks, labels = self.anomaly_generator(
            input_features=None,
            adapted_features=adapted,
            masks=masks,
            labels=labels
        )
        
        # 判别器预测
        anomaly_map, score = self.segdec(noised_adapt)
        return anomaly_map, score, masks, labels
    
    # 推理时：直接预测
    anomaly_map, score = self.segdec(adapted)
    anomaly_map = self.anomaly_map_generator(anomaly_map, images.shape[-2:])
    return InferenceBatch(pred_score=score, anomaly_map=anomaly_map)
```

#### 12.3 异常生成器

**代码来源**：`anomalib/models/image/supersimplenet/anomaly_generator.py`

```python
class AnomalyGenerator:
    """使用Perlin噪声生成异常特征"""
    def __init__(self, noise_mean=0, noise_std=0.015, threshold=0.2):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.threshold = threshold
    
    def __call__(self, adapted_features, masks, labels):
        """生成异常特征"""
        # 生成Perlin噪声
        noise = generate_perlin_noise(adapted_features.shape)
        
        # 应用噪声到特征
        noised_features = adapted_features + noise * masks
        
        return adapted_features, noised_features, masks, labels
```

**关键特点**：
- 使用Perlin噪声生成自然的异常模式
- 只在mask区域添加噪声，保持正常区域不变
- 噪声强度可控制（noise_std参数）


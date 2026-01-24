# FastFlow - Unsupervised Anomaly Detection and Localization 完整算法解读

## 论文信息
- **标题**: FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows
- **年份**: 2021
- **论文链接**: https://arxiv.org/abs/2111.07677
- **核心思想**: 使用2D归一化流（Normalizing Flow）作为概率分布估计器，在预训练特征图上学习正常数据的分布

## 算法概述

FastFlow是一种基于归一化流的异常检测方法。它使用2D归一化流在预训练特征图上学习正常数据的分布，通过计算负对数似然来检测异常。核心创新是设计了一种高效的2D归一化流架构，保持特征图的空间结构。

### 整体架构

```
输入图像 → 预训练特征提取 → LayerNorm → 2D归一化流 → 负对数似然 → 异常图生成
```

## 一、特征提取阶段

### 1.1 预训练特征提取器

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第129-162行)

FastFlow支持多种backbone：ResNet、Wide ResNet、Vision Transformer (ViT)、CaiT。

```python
class FastflowModel(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        
        # 支持多种backbone
        if backbone in {"cait_m48_448", "deit_base_distilled_patch16_384"}:
            # Vision Transformer
            self.feature_extractor = timm.create_model(backbone, pretrained=pre_trained)
            channels = [768]
            scales = [16]
        elif backbone in {"resnet18", "wide_resnet50_2"}:
            # CNN backbone
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                out_indices=[1, 2, 3],  # 提取layer1, layer2, layer3
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()
            
            # 为CNN特征添加LayerNorm（Transformer使用预训练的norm）
            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales, strict=True):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    ),
                )
        
        # 冻结特征提取器参数
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
```

**关键设计**：
- **多尺度特征**：从多个中间层提取特征（layer1, layer2, layer3）
- **LayerNorm**：对CNN特征应用LayerNorm，稳定训练
- **参数冻结**：特征提取器参数冻结，只训练归一化流

### 1.2 特征提取方法

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第220-274行)

```python
def _get_cnn_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
    """提取CNN特征并应用LayerNorm"""
    features = self.feature_extractor(input_tensor)
    # features是tuple，转换为list
    if not isinstance(features, list):
        features = list(features)
    
    # 应用LayerNorm
    normalized_features = []
    for feature, norm in zip(features, self.norms, strict=True):
        normalized_features.append(norm(feature))
    
    return normalized_features

def _get_vit_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
    """提取ViT特征（使用CLS token）"""
    # ViT特征提取逻辑
    # ...
    
def _get_cait_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
    """提取CaiT特征"""
    # CaiT特征提取逻辑
    # ...
```

## 二、归一化流架构

### 2.1 归一化流基本原理

归一化流是一种可逆的生成模型，通过一系列可逆变换将简单分布（如标准高斯）转换为复杂分布。关键优势是可以精确计算概率密度。

**数学原理**：
- 变换：z = f(x)，其中f是可逆函数
- 概率密度：p_X(x) = p_Z(f(x)) |det(∂f/∂x)|
- 对数似然：log p_X(x) = log p_Z(f(x)) + log |det(∂f/∂x)|

### 2.2 FastFlow块创建

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第60-92行)

FastFlow使用FrEIA库的`SequenceINN`和`AllInOneBlock`构建归一化流。

```python
def create_fast_flow_block(
    input_dimensions: list[int],  # [C, H, W]
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    """创建FastFlow归一化流块
    
    基于论文Figure 2和Section 3.3。
    每个块包含多个flow steps，交替使用conv1x1和conv3x3。
    
    Args:
        input_dimensions: [C, H, W] - 输入特征维度
        conv3x3_only: 是否只使用conv3x3（False时交替使用conv1x1和conv3x3）
        hidden_ratio: 隐藏层通道数比例
        flow_steps: 流步骤数（默认8）
        clamp: 仿射参数裁剪范围（默认2.0）
    
    Returns:
        SequenceINN: FastFlow归一化流块
    """
    nodes = SequenceINN(*input_dimensions)
    
    for i in range(flow_steps):
        # 交替使用conv1x1和conv3x3
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    
    return nodes
```

**设计说明**：
- **交替卷积**：奇数步使用conv1x1，偶数步使用conv3x3（如果`conv3x3_only=False`）
- **AllInOneBlock**：FrEIA库提供的完整归一化流块，包含可逆卷积和仿射耦合层
- **affine_clamping**：限制仿射参数范围，提升训练稳定性

### 2.3 子网络构造（Subnet）

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第27-57行)

子网络用于预测仿射耦合层的参数（scale和shift）。

```python
def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """子网络构造函数
    
    返回一个函数，用于创建预测仿射参数的卷积网络。
    
    Args:
        kernel_size: 卷积核大小（1或3）
        hidden_ratio: 隐藏层通道数比例
    
    Returns:
        Callable: 函数f(channels_in, channels_out) -> nn.Sequential
    """
    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        # 计算隐藏层通道数
        hidden_channels = int(in_channels * hidden_ratio)
        
        # 手动计算padding（因为padding="same"在ONNX导出时有问题）
        padding_dims = (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
        padding = (*padding_dims, *padding_dims)
        
        return nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.ReLU(),
            nn.ZeroPad2d(padding),
            nn.Conv2d(hidden_channels, out_channels, kernel_size),
        )
    
    return subnet_conv
```

**网络结构**：
- 输入：特征图的一部分（用于预测另一部分的仿射参数）
- 隐藏层：ReLU激活的卷积层
- 输出：仿射参数（scale和shift）

### 2.4 归一化流块初始化

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第163-172行)

为每个特征层创建独立的归一化流块。

```python
self.fast_flow_blocks = nn.ModuleList()
for channel, scale in zip(channels, scales, strict=True):
    self.fast_flow_blocks.append(
        create_fast_flow_block(
            input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
            conv3x3_only=conv3x3_only,
            hidden_ratio=hidden_ratio,
            flow_steps=flow_steps,
        ),
    )
```

**设计**：
- 每个特征层有独立的归一化流块
- 输入维度根据特征图尺寸自动计算
- 保持特征图的空间结构

## 三、前向传播

### 3.1 前向传播流程

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第175-210行)

```python
def forward(
    self, 
    input_tensor: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
    """前向传播
    
    Args:
        input_tensor: [B, 3, H, W] - 输入图像
    
    Returns:
        训练时: (hidden_variables, log_jacobians)
        推理时: InferenceBatch(pred_score, anomaly_map)
    """
    self.feature_extractor.eval()
    
    # 1. 提取特征（根据backbone类型选择方法）
    if isinstance(self.feature_extractor, VisionTransformer):
        features = self._get_vit_features(input_tensor)
    elif isinstance(self.feature_extractor, Cait):
        features = self._get_cait_features(input_tensor)
    else:
        features = self._get_cnn_features(input_tensor)
    
    # 2. 通过归一化流变换（计算隐藏变量和雅可比行列式）
    hidden_variables: list[torch.Tensor] = []
    log_jacobians: list[torch.Tensor] = []
    
    for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
        # 归一化流变换：x -> z
        hidden_variable, log_jacobian = fast_flow_block(feature)
        hidden_variables.append(hidden_variable)
        log_jacobians.append(log_jacobian)
    
    if self.training:
        # 训练时：返回隐藏变量和雅可比行列式
        return hidden_variables, log_jacobians
    
    # 推理时：生成异常图
    anomaly_map = self.anomaly_map_generator(hidden_variables)
    pred_score = torch.amax(anomaly_map, dim=(-2, -1))
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map.squeeze(1))
```

**关键步骤**：
1. **特征提取**：根据backbone类型提取多尺度特征
2. **归一化流变换**：每个特征层通过独立的归一化流块
3. **输出**：训练时返回隐藏变量和雅可比行列式，推理时生成异常图

## 四、训练阶段

### 4.1 损失函数

**代码来源**：`anomalib/models/image/fastflow/loss.py` (第28-65行)

FastFlow使用负对数似然作为损失函数，目标是最大化正常数据的似然。

**数学公式**（论文Equation 3）：
```
L = -log p(x) = -log p(z) - log |det(∂f/∂x)|
```

其中：
- z = f(x) 是归一化流变换后的隐藏变量
- p(z) 是标准高斯分布 N(0, I)
- log |det(∂f/∂x)| 是雅可比行列式的对数

```python
class FastflowLoss(nn.Module):
    """FastFlow损失函数：负对数似然"""
    
    @staticmethod
    def forward(
        hidden_variables: list[torch.Tensor],  # 隐藏变量列表
        jacobians: list[torch.Tensor]           # 雅可比行列式对数列表
    ) -> torch.Tensor:
        """计算负对数似然损失
        
        Args:
            hidden_variables: 每个元素是 [B, C, H, W] - 归一化流变换后的特征
            jacobians: 每个元素是 [B] - 雅可比行列式的对数
        
        Returns:
            标量损失值
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            # 计算负对数似然：
            # -log p(z) = 0.5 * ||z||^2 (标准高斯分布)
            # 减去雅可比行列式（变换的贡献）
            loss += torch.mean(
                0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian
            )
        
        return loss
```

**损失计算说明**：
- **0.5 * ||z||²**：标准高斯分布的负对数似然
- **- jacobian**：减去雅可比行列式（因为变量变换）
- **多尺度融合**：对所有特征层的损失求和

### 4.2 训练流程

**代码来源**：`anomalib/models/image/fastflow/lightning_model.py`

```python
def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
    """训练步骤"""
    # 前向传播
    hidden_vars, log_jacobians = self.model(batch.image)
    
    # 计算损失
    loss = self.loss(hidden_vars, log_jacobians)
    
    return {"loss": loss}
```

**训练特点**：
- 只使用正常样本训练
- 最大化正常数据的似然（最小化负对数似然）
- 归一化流学习将正常特征映射到标准高斯分布

## 五、推理阶段（异常检测）

### 5.1 异常图生成

**代码来源**：`anomalib/models/image/fastflow/anomaly_map.py` (第23-84行)

FastFlow通过计算负对数概率生成异常图。

```python
class AnomalyMapGenerator(nn.Module):
    """从隐藏变量生成异常热力图"""
    
    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)
    
    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        """生成异常热力图
        
        流程：
        1. 对每个隐藏变量计算负对数概率
        2. 转换为概率（通过指数）
        3. 上采样到输入图像尺寸
        4. 平均所有flow maps
        
        Args:
            hidden_variables: 每个元素是 [B, C, H, W] - 归一化流变换后的特征
        
        Returns:
            [B, 1, H, W] - 异常热力图
        """
        flow_maps: list[torch.Tensor] = []
        
        for hidden_variable in hidden_variables:
            # 1. 计算负对数概率：-log p(z) = 0.5 * ||z||^2
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            
            # 2. 转换为概率
            prob = torch.exp(log_prob)
            
            # 3. 上采样到输入图像尺寸
            flow_map = F.interpolate(
                input=-prob,  # 使用负概率作为异常分数
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        
        # 4. 堆叠并平均所有flow maps
        flow_maps = torch.stack(flow_maps, dim=-1)  # [B, 1, H, W, num_scales]
        return torch.mean(flow_maps, dim=-1)       # [B, 1, H, W]
```

**异常分数计算**：
- **负对数概率**：`-log p(z) = 0.5 * ||z||²`，值越大表示越异常
- **概率转换**：通过指数转换为概率，然后取负值作为异常分数
- **多尺度融合**：对所有特征层的异常图求平均

### 5.2 完整推理流程

```python
# 1. 加载模型
model = Fastflow.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数（最大值）
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

**推理步骤**：
1. 提取测试图像的多尺度特征
2. 通过归一化流变换得到隐藏变量
3. 计算每个位置的负对数概率
4. 上采样到原始图像尺寸
5. 融合多尺度异常图
6. 返回异常分数和异常图

#### 5.2 异常定位
- 异常分数图直接提供像素级定位
- 可以应用阈值进行二值化
- 使用后处理（如形态学操作）优化结果

### 6. 关键实现细节

#### 6.1 数值稳定性
- 使用tanh限制仿射参数范围
- 添加小的正则化项避免数值问题
- 使用混合精度训练提升稳定性

#### 6.2 多尺度融合
- 不同特征层捕获不同尺度的信息
- 可以学习不同层的融合权重
- 或使用简单的平均/最大值融合

#### 6.3 计算效率
- 归一化流的前向和反向都很快
- 可以并行处理多个特征层
- 推理时只需要前向传播

## 六、总结
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        # 特征提取器（冻结参数）
        if backbone in {"resnet18", "wide_resnet50_2"}:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()
            
            # 为每个特征层添加LayerNorm
            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )
        
        # 冻结特征提取器
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        
        # 为每个特征层创建FastFlow块
        self.fast_flow_blocks = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
```

**架构特点**：
- 支持ResNet和Vision Transformer backbone
- 每个特征层使用独立的归一化流块
- LayerNorm归一化特征，提升训练稳定性

#### 11.4 前向传播

**代码来源**：`anomalib/models/image/fastflow/torch_model.py` (第175-209行)

```python
def forward(self, input_tensor: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
    """前向传播"""
    self.feature_extractor.eval()
    
    # 提取特征
    if isinstance(self.feature_extractor, VisionTransformer):
        features = self._get_vit_features(input_tensor)
    elif isinstance(self.feature_extractor, Cait):
        features = self._get_cait_features(input_tensor)
    else:
        features = self._get_cnn_features(input_tensor)
    
    # 通过归一化流
    hidden_variables: list[torch.Tensor] = []
    log_jacobians: list[torch.Tensor] = []
    
    for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
        hidden_variable, log_jacobian = fast_flow_block(feature)
        hidden_variables.append(hidden_variable)
        log_jacobians.append(log_jacobian)
    
    if self.training:
        return hidden_variables, log_jacobians
    
    # 推理时生成异常图
    anomaly_map = self.anomaly_map_generator(hidden_variables)
    pred_score = torch.amax(anomaly_map, dim=(-2, -1))
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
```



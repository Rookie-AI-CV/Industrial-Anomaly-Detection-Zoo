# DRÆM - Discriminatively Trained Reconstruction Embedding 完整算法解读

## 论文信息
- **标题**: DRÆM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection
- **年份**: 2021
- **论文链接**: https://arxiv.org/abs/2108.07610
- **核心思想**: 使用判别式训练的重建嵌入，通过重建正常样本和合成异常样本，训练能够区分正常和异常的双网络模型

## 算法概述

DRÆM是一种基于重建和判别的异常检测方法。它包含两个子网络：
1. **重建子网络**：学习重建输入图像（包括正常和异常区域）
2. **判别子网络**：比较原始图像和重建图像，预测异常掩码

### 整体架构

```
正常图像 + 合成异常图像 → 重建子网络 → 重建图像 → 拼接 → 判别子网络 → 异常掩码
```

## 一、模型架构

### 1.1 双网络结构

**代码来源**：`anomalib/models/image/draem/torch_model.py` (第25-73行)

DRÆM包含两个独立的子网络，分别负责重建和判别。

```python
class DraemModel(nn.Module):
    """DRÆM模型：重建子网络 + 判别子网络"""
    
    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        # 重建子网络（自编码器）
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        
        # 判别子网络（输入是原始图像和重建图像的拼接）
        self.discriminative_subnetwork = DiscriminativeSubNetwork(
            in_channels=6,  # 3 (原始) + 3 (重建)
            out_channels=2  # 正常/异常二分类
        )
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple | InferenceBatch:
        """前向传播
        
        Args:
            batch: [B, 3, H, W] - 输入图像（可能包含合成异常）
        
        Returns:
            训练时: (reconstruction, prediction)
            推理时: InferenceBatch(pred_score, anomaly_map)
        """
        # 1. 重建子网络：重建图像
        reconstruction = self.reconstructive_subnetwork(batch)
        
        # 2. 拼接原始图像和重建图像
        concatenated_inputs = torch.cat([batch, reconstruction], axis=1)  # [B, 6, H, W]
        
        # 3. 判别子网络：预测异常掩码
        prediction = self.discriminative_subnetwork(concatenated_inputs)  # [B, 2, H, W]
        
        if self.training:
            return reconstruction, prediction
        
        # 推理时：生成异常图
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]  # 取异常类概率
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
```

### 1.2 重建子网络

**代码来源**：`anomalib/models/image/draem/torch_model.py` (第76-116行)

重建子网络使用U-Net风格的编码器-解码器架构。

```python
class ReconstructiveSubNetwork(nn.Module):
    """重建子网络（自编码器）"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_width: int = 128,
        sspcab: bool = False,
    ) -> None:
        super().__init__()
        # 编码器
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        
        # 解码器
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """编码并重建图像"""
        encoded = self.encoder(batch)
        return self.decoder(encoded)
```

**编码器结构**（EncoderReconstructive）：
- 多个卷积块，逐步下采样
- 使用SSPCAB（可选）增强特征表示
- 输出编码特征

**解码器结构**（DecoderReconstructive）：
- 多个转置卷积块，逐步上采样
- 恢复到原始图像尺寸
- 输出重建图像

### 1.3 判别子网络

**代码来源**：`anomalib/models/image/draem/torch_model.py` (第119-155行)

判别子网络使用带跳跃连接的U-Net架构，比较原始图像和重建图像。

```python
class DiscriminativeSubNetwork(nn.Module):
    """判别子网络：预测异常掩码"""
    
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base_width: int = 64) -> None:
        super().__init__()
        # 编码器（带跳跃连接）
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        
        # 解码器（使用跳跃连接）
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """生成异常掩码预测
        
        Args:
            batch: [B, 6, H, W] - 拼接的原始图像和重建图像
        
        Returns:
            [B, 2, H, W] - 每个像素的正常/异常类分数
        """
        # 编码器返回多个中间特征（用于跳跃连接）
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        
        # 解码器使用跳跃连接重建异常掩码
        return self.decoder_segment(act1, act2, act3, act4, act5, act6)
```

**关键设计**：
- **输入拼接**：原始图像和重建图像拼接，提供对比信息
- **跳跃连接**：保留细节信息，提升异常定位精度
- **二分类输出**：每个像素输出正常/异常两类分数

## 二、损失函数

### 2.1 多组件损失函数

**代码来源**：`anomalib/models/image/draem/loss.py` (第26-74行)

DRÆM使用组合损失函数，包含三个组件：L2重建损失、SSIM损失和Focal损失。

```python
class DraemLoss(nn.Module):
    """DRÆM损失函数：L2 + SSIM + Focal Loss"""
    
    def __init__(self) -> None:
        super().__init__()
        # L2重建损失（MSE）
        self.l2_loss = nn.modules.loss.MSELoss()
        
        # Focal Loss（用于异常掩码预测，处理类别不平衡）
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        
        # SSIM损失（结构相似性，保持图像结构）
        self.ssim_loss = SSIMLoss(window_size=11)
    
    def forward(
        self,
        input_image: torch.Tensor,      # [B, 3, H, W] - 原始图像
        reconstruction: torch.Tensor,    # [B, 3, H, W] - 重建图像
        anomaly_mask: torch.Tensor,      # [B, 1, H, W] - 异常掩码（GT）
        prediction: torch.Tensor,        # [B, 2, H, W] - 预测的异常掩码
    ) -> torch.Tensor:
        """计算组合损失
        
        Args:
            input_image: 原始输入图像
            reconstruction: 重建子网络的输出
            anomaly_mask: 真实异常掩码（合成异常的位置）
            prediction: 判别子网络的输出（正常/异常类分数）
        
        Returns:
            标量损失值
        """
        # 1. L2重建损失：重建图像与原始图像的差异
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        
        # 2. Focal Loss：异常掩码预测损失（处理类别不平衡）
        focal_loss_val = self.focal_loss(
            prediction, 
            anomaly_mask.squeeze(1).long()  # [B, H, W]
        )
        
        # 3. SSIM损失：结构相似性损失（权重为2）
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        
        # 总损失
        return l2_loss_val + ssim_loss_val + focal_loss_val
```

**损失组件说明**：
- **L2损失**：衡量重建图像与原始图像的像素级差异
- **SSIM损失**：保持图像的结构相似性，权重为2
- **Focal Loss**：处理正常/异常像素的类别不平衡问题

### 2.2 合成异常生成

**注意**：DRÆM的合成异常生成通常在数据加载器或训练循环中实现，使用Perlin噪声等方法。Anomalib的实现中，这部分可能在数据增强模块中。

## 三、训练阶段

### 3.1 训练流程

**代码来源**：`anomalib/models/image/draem/lightning_model.py`

```python
def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
    """训练步骤"""
    # batch.image 可能包含合成异常
    # batch.mask 是异常掩码（如果有）
    
    # 前向传播
    reconstruction, prediction = self.model(batch.image)
    
    # 计算损失
    loss = self.loss(
        input_image=batch.image,
        reconstruction=reconstruction,
        anomaly_mask=batch.mask,  # 合成异常的掩码
        prediction=prediction
    )
    
    return {"loss": loss}
```

**训练特点**：
- 使用正常图像和合成异常图像训练
- 重建子网络学习重建所有区域（包括异常）
- 判别子网络学习区分正常和异常区域
- 通过Focal Loss处理类别不平衡

## 四、推理阶段（异常检测）

### 4.1 异常检测流程

**代码来源**：`anomalib/models/image/draem/torch_model.py` (第65-73行)

推理时，模型直接输出异常掩码预测。

```python
# 在forward方法中（推理模式）
if not self.training:
    # 1. 重建图像
    reconstruction = self.reconstructive_subnetwork(batch)
    
    # 2. 拼接原始和重建图像
    concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
    
    # 3. 判别子网络预测异常掩码
    prediction = self.discriminative_subnetwork(concatenated_inputs)
    
    # 4. 使用softmax获取异常类概率
    anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]  # [B, H, W]
    
    # 5. 图像级异常分数（最大值）
    pred_score = torch.amax(anomaly_map, dim=(-2, -1))
    
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
```

**推理步骤**：
1. 重建子网络重建测试图像
2. 拼接原始图像和重建图像
3. 判别子网络预测每个像素的正常/异常概率
4. 使用softmax获取异常类概率作为异常图
5. 图像级分数取异常图的最大值

### 4.2 完整推理示例

```python
# 1. 加载模型
model = Draem.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数
anomaly_map = predictions.anomaly_map    # 像素级异常图 [B, H, W]
```

## 五、关键实现细节

### 5.1 编码器-解码器架构

**重建子网络编码器**（EncoderReconstructive）：
- 多个卷积块，逐步下采样
- 可选SSPCAB（Spatial Pyramid Convolutional Attention Block）增强特征
- 输出编码特征用于解码

**重建子网络解码器**（DecoderReconstructive）：
- 转置卷积块，逐步上采样
- 恢复到原始图像尺寸
- 输出重建图像

**判别子网络编码器**（EncoderDiscriminative）：
- 6个卷积块，逐步下采样
- 返回多个中间特征（用于跳跃连接）
- 输出6个激活图（act1-act6）

**判别子网络解码器**（DecoderDiscriminative）：
- 使用跳跃连接融合多尺度特征
- 逐步上采样到原始尺寸
- 输出2通道特征图（正常/异常类分数）

### 5.2 跳跃连接机制

判别子网络使用跳跃连接，保留细节信息：
- 编码器的中间特征直接传递到解码器
- 提升异常定位的精度
- 类似U-Net的架构设计

## 六、算法优缺点分析

### 6.1 优点

1. **双网络设计**：重建和判别分离，各司其职
2. **合成异常训练**：无需真实异常样本，只需正常样本
3. **多组件损失**：L2 + SSIM + Focal Loss，全面优化
4. **像素级定位**：直接输出异常掩码，无需后处理

### 6.2 缺点

1. **需要合成异常**：依赖异常生成的质量
2. **训练复杂**：需要同时训练两个网络
3. **计算量大**：两个网络的前向传播和反向传播
4. **合成异常限制**：可能无法覆盖所有真实异常类型

## 七、关键超参数

### 7.1 模型超参数

- **base_width**：重建子网络的基础宽度（默认128）
- **discriminative_base_width**：判别子网络的基础宽度（默认64）
- **sspcab**：是否使用SSPCAB增强（默认False）

### 7.2 损失函数超参数

- **SSIM权重**：2.0（在损失函数中）
- **Focal Loss alpha**：1.0
- **Focal Loss reduction**：mean

## 八、使用示例

### 8.1 基本使用

```python
from anomalib.models import Draem
from anomalib.data import MVTecAD
from anomalib.engine import Engine

# 初始化模型
model = Draem(
    sspcab=False,  # 是否使用SSPCAB
)

# 加载数据（需要合成异常生成）
datamodule = MVTecAD(category="bottle")

# 训练
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# 推理
predictions = engine.predict(model=model, datamodule=datamodule)
```

## 九、总结

DRÆM是一种基于重建和判别的异常检测方法，核心思想是：
1. **重建子网络**：学习重建包含异常的图像
2. **判别子网络**：比较原始和重建图像，预测异常掩码
3. **合成异常训练**：使用Perlin噪声等方法生成训练异常
4. **多组件损失**：L2 + SSIM + Focal Loss全面优化

该方法在表面缺陷检测任务上表现优异，特别适合工业异常检测场景。

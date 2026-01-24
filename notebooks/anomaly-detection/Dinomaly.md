# Dinomaly - The Less Is More Philosophy 完整算法解读

## 论文信息
- **标题**: Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection
- **年份**: 2024
- **论文链接**: https://arxiv.org/abs/2401.07557
- **核心思想**: 采用"少即是多"的哲学，使用DINOv2编码器-瓶颈-解码器架构，通过特征重建误差检测异常

## 算法概述

Dinomaly是一种基于Vision Transformer的异常检测方法。它使用DINOv2作为编码器提取特征，通过瓶颈层压缩后，解码器重建特征。正常样本能很好地重建，异常样本重建困难，产生较大误差。核心创新是"少即是多"的设计哲学，简化架构和训练过程。

### 整体架构

```
输入图像 → DINOv2编码器 → 中间层特征提取 → 瓶颈层 → 解码器 → 特征重建 → 余弦相似度 → 异常图生成
```

## 一、模型架构

### 1.1 编码器-瓶颈-解码器结构

**代码来源**：`anomalib/models/image/dinomaly/torch_model.py` (第58-189行)

Dinomaly使用DINOv2作为编码器，提取中间层特征，通过瓶颈层压缩后，解码器重建。

```python
class DinomalyModel(nn.Module):
    """Dinomaly模型：DINOv2编码器 + 瓶颈层 + 解码器"""
    
    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
    ) -> None:
        super().__init__()
        
        # 1. DINOv2编码器（预训练，冻结）
        encoder = DinoV2Loader(vit_factory=dinomaly_vision_transformer).load(encoder_name)
        self.encoder = encoder
        
        # 获取架构配置
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]  # 768 for base
        num_heads = arch_config["num_heads"]   # 12 for base
        target_layers = arch_config["target_layers"]  # [2,3,4,5,6,7,8,9]
        
        # 2. 瓶颈层：MLP压缩特征
        bottleneck = []
        bottle_neck_mlp = DinomalyMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,  # 768 -> 3072 -> 768
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=bottleneck_dropout,
            bias=False,
            apply_input_dropout=True,
        )
        bottleneck.append(bottle_neck_mlp)
        self.bottleneck = nn.ModuleList(bottleneck)
        
        # 3. 解码器：Vision Transformer块重建特征
        decoder = []
        for _ in range(decoder_depth):
            decoder_block = DecoderViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn_drop=0.0,
                attn=LinearAttention,  # 线性注意力（简化）
            )
            decoder.append(decoder_block)
        self.decoder = nn.ModuleList(decoder)
        
        # 配置
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder or [[0,1,2,3], [4,5,6,7]]
        self.fuse_layer_decoder = fuse_layer_decoder or [[0,1,2,3], [4,5,6,7]]
        self.remove_class_token = remove_class_token
        
        # 高斯模糊（异常图平滑）
        self.gaussian_blur = GaussianBlur2d(
            sigma=4,
            channels=1,
            kernel_size=5,
        )
        
        # 损失函数：余弦硬挖掘损失
        self.loss_fn = CosineHardMiningLoss()
```

**关键设计**：
- **DINOv2编码器**：使用预训练的DINOv2 ViT，提取通用特征
- **中间层特征**：从编码器的中间层（如layer 2-9）提取特征
- **瓶颈层**：MLP压缩特征，学习紧凑表示
- **解码器**：Vision Transformer块重建特征
- **线性注意力**：使用LinearAttention简化计算

### 1.2 特征提取与重建

**代码来源**：`anomalib/models/image/dinomaly/torch_model.py` (第191-243行)

```python
def get_encoder_decoder_outputs(
    self, 
    x: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """提取编码器和解码器特征
    
    Args:
        x: [B, 3, H, W] - 输入图像
    
    Returns:
        (encoder_features, decoder_features)
        - encoder_features: 编码器特征列表
        - decoder_features: 解码器特征列表
    """
    # 1. 编码器提取特征
    encoder_features = []
    with torch.no_grad():
        # 使用forward hook提取中间层特征
        def hook_fn(module, input, output):
            encoder_features.append(output)
        
        hooks = []
        for layer_idx in self.target_layers:
            hook = self.encoder.blocks[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # 前向传播
        _ = self.encoder(x)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
    
    # 2. 处理特征（移除class token等）
    processed_features = []
    for feat in encoder_features:
        if self.remove_class_token:
            # 移除class token，只保留patch tokens
            feat = feat[:, 1:, :]  # [B, N_patches, D]
        processed_features.append(feat)
    
    # 3. 融合特征组（将多个层融合）
    fused_encoder_features = []
    for layer_group in self.fuse_layer_encoder:
        group_features = [processed_features[i] for i in layer_group]
        fused = self._fuse_feature(group_features)  # 平均融合
        fused_encoder_features.append(fused)
    
    # 4. 瓶颈层压缩
    bottleneck_outputs = []
    for feat in fused_encoder_features:
        compressed = self.bottleneck[0](feat)  # [B, N, D]
        bottleneck_outputs.append(compressed)
    
    # 5. 解码器重建
    decoder_features = []
    for compressed in bottleneck_outputs:
        reconstructed = compressed
        for decoder_block in self.decoder:
            reconstructed = decoder_block(reconstructed)
        decoder_features.append(reconstructed)
    
    # 6. 融合解码器特征组
    fused_decoder_features = []
    for i, layer_group in enumerate(self.fuse_layer_decoder):
        group_features = [decoder_features[j] for j in layer_group]
        fused = self._fuse_feature(group_features)
        fused_decoder_features.append(fused)
    
    return fused_encoder_features, fused_decoder_features

@staticmethod
def _fuse_feature(features: list[torch.Tensor]) -> torch.Tensor:
    """融合多个特征（平均）"""
    return torch.stack(features).mean(dim=0)
```

**关键步骤**：
1. **特征提取**：使用forward hook提取编码器中间层特征
2. **特征融合**：将多个层融合为特征组（如[0,1,2,3]和[4,5,6,7]）
3. **瓶颈压缩**：通过MLP压缩特征
4. **解码重建**：通过解码器重建特征
5. **特征对齐**：编码器和解码器特征对应，便于计算差异
## 二、异常图计算

### 2.1 余弦相似度异常图

**代码来源**：`anomalib/models/image/dinomaly/torch_model.py` (第300-337行)

Dinomaly使用余弦相似度计算编码器和解码器特征的差异，生成异常图。

```python
@staticmethod
def calculate_anomaly_maps(
    source_feature_maps: list[torch.Tensor],  # 编码器特征列表
    target_feature_maps: list[torch.Tensor],  # 解码器特征列表
    out_size: int | tuple[int, int] = 392,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """计算异常图：通过比较编码器和解码器特征
    
    Args:
        source_feature_maps: 编码器特征列表（每个元素是 [B, N, D]）
        target_feature_maps: 解码器特征列表（每个元素是 [B, N, D]）
        out_size: 输出异常图尺寸
    
    Returns:
        (anomaly_map, anomaly_map_list)
        - anomaly_map: [B, 1, H, W] - 融合后的异常图
        - anomaly_map_list: 每个特征组的异常图列表
    """
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    
    anomaly_map_list = []
    
    # 对每个特征组计算异常图
    for i in range(len(target_feature_maps)):
        fs = source_feature_maps[i]  # 编码器特征 [B, N, D]
        ft = target_feature_maps[i]  # 解码器特征 [B, N, D]
        
        # 计算余弦不相似度：1 - cosine_similarity
        a_map = 1 - F.cosine_similarity(fs, ft, dim=-1)  # [B, N]
        
        # 重塑为空间图
        a_map = torch.unsqueeze(a_map, dim=1)  # [B, 1, N]
        
        # 上采样到目标尺寸
        a_map = F.interpolate(
            a_map, 
            size=out_size, 
            mode="bilinear", 
            align_corners=True
        )
        
        anomaly_map_list.append(a_map)
    
    # 融合所有特征组的异常图（平均）
    anomaly_map = torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True)
    
    return anomaly_map, anomaly_map_list
```

**异常图计算说明**：
- **余弦不相似度**：`1 - cosine_similarity`，值越大表示差异越大
- **多尺度融合**：对所有特征组的异常图求平均
- **空间重塑**：将patch tokens重塑为空间图

### 2.2 前向传播与异常分数计算

**代码来源**：`anomalib/models/image/dinomaly/torch_model.py` (第245-298行)

```python
def forward(
    self, 
    batch: torch.Tensor, 
    global_step: int | None = None
) -> torch.Tensor | InferenceBatch:
    """前向传播
    
    Args:
        batch: [B, 3, H, W] - 输入图像
        global_step: 训练步数（用于损失计算）
    
    Returns:
        训练时: 损失值
        推理时: InferenceBatch(pred_score, anomaly_map)
    """
    # 1. 提取编码器和解码器特征
    en, de = self.get_encoder_decoder_outputs(batch)
    image_size = (batch.shape[2], batch.shape[3])
    
    if self.training:
        if global_step is None:
            raise ValueError("global_step must be provided during training")
        
        # 训练时：计算损失
        return self.loss_fn(
            encoder_features=en, 
            decoder_features=de, 
            global_step=global_step
        )
    
    # 推理时：计算异常图
    anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)
    anomaly_map_resized = anomaly_map.clone()
    
    # 调整尺寸（如果设置了DEFAULT_RESIZE_SIZE）
    if DEFAULT_RESIZE_SIZE is not None:
        anomaly_map = F.interpolate(
            anomaly_map, 
            size=DEFAULT_RESIZE_SIZE, 
            mode="bilinear", 
            align_corners=False
        )
    
    # 高斯模糊平滑
    anomaly_map = self.gaussian_blur(anomaly_map)
    
    # 计算图像级异常分数
    if DEFAULT_MAX_RATIO == 0:
        # 使用最大值
        sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
    else:
        # 使用前MAX_RATIO%的最大值平均
        anomaly_map_flat = anomaly_map.flatten(1)
        top_k = int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO)
        sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :top_k]
        sp_score = sp_score.mean(dim=1)
    
    pred_score = sp_score
    
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map_resized)
```

**异常分数计算**：
- **最大值**：如果`DEFAULT_MAX_RATIO=0`，使用异常图的最大值
- **Top-K平均**：否则使用前K%的最大值平均（更稳定）

## 三、损失函数

### 3.1 余弦硬挖掘损失

**代码来源**：`anomalib/models/image/dinomaly/components/loss.py` (第15-141行)

Dinomaly使用余弦硬挖掘损失，这是一种特殊的训练策略，防止解码器过度重建异常区域。

```python
class CosineHardMiningLoss(torch.nn.Module):
    """余弦硬挖掘损失
    
    核心思想：降低重建良好的点（easy points）的梯度贡献，
    使训练专注于难以重建的点（hard points），防止解码器学习重建异常模式。
    
    Args:
        p_final: 最终要降低梯度的easy points比例（默认0.9，即90%）
        p_schedule_steps: p值的调度步数（默认1000）
        factor: 梯度降低因子（默认0.1，即降低到10%）
    """
    
    def __init__(
        self, 
        p_final: float = 0.9, 
        p_schedule_steps: int = 1000, 
        factor: float = 0.1
    ) -> None:
        super().__init__()
        self.p_final = p_final
        self.factor = factor
        self.p_schedule_steps = p_schedule_steps
        self.p = 0.0  # 动态更新
    
    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            encoder_features: 编码器特征列表
            decoder_features: 解码器特征列表
            global_step: 当前训练步数
        
        Returns:
            标量损失值
        """
        # 更新p值（调度）
        self._update_p_schedule(global_step)
        
        cos_loss = torch.nn.CosineSimilarity()
        loss = torch.tensor(0.0, device=encoder_features[0].device)
        
        for item in range(len(encoder_features)):
            en_ = encoder_features[item].detach()  # 冻结编码器梯度
            de_ = decoder_features[item]
            
            # 1. 计算每个点的余弦不相似度
            with torch.no_grad():
                point_dist = 1 - cos_loss(en_, de_).unsqueeze(1)  # [B, 1, H, W]
            
            # 2. 找到重建良好的点（easy points）
            # 选择前(1-p)比例的点作为hard points
            k = max(1, int(point_dist.numel() * (1 - self.p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]  # 阈值
            
            # 3. 计算损失（所有点）
            loss += torch.mean(
                1 - cos_loss(
                    en_.reshape(en_.shape[0], -1), 
                    de_.reshape(de_.shape[0], -1)
                )
            )
            
            # 4. 降低easy points的梯度贡献
            partial_func = partial(
                self._modify_grad,
                indices_to_modify=point_dist < thresh,  # easy points
                gradient_multiply_factor=self.factor,   # 降低到10%
            )
            de_.register_hook(partial_func)
        
        return loss / len(encoder_features)
    
    @staticmethod
    def _modify_grad(
        x: torch.Tensor,
        indices_to_modify: torch.Tensor,
        gradient_multiply_factor: float = 0.1,
    ) -> torch.Tensor:
        """修改梯度：降低easy points的梯度贡献"""
        indices_to_modify = indices_to_modify.expand_as(x)
        result = x.clone()
        result[indices_to_modify] = result[indices_to_modify] * gradient_multiply_factor
        return result
    
    def _update_p_schedule(self, global_step: int) -> None:
        """更新p值调度"""
        self.p = min(
            self.p_final * global_step / self.p_schedule_steps, 
            self.p_final
        )
```

**损失函数说明**：
- **硬挖掘**：降低重建良好的点（easy points）的梯度，专注于难以重建的点
- **防止过拟合**：防止解码器学习重建异常模式
- **p值调度**：训练过程中逐渐增加p值，从0到0.9

## 四、完整训练与推理流程

### 4.1 训练流程

**代码来源**：`anomalib/models/image/dinomaly/lightning_model.py`

```python
def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
    """训练步骤"""
    # 前向传播（返回损失）
    loss = self.model(batch.image, global_step=self.global_step)
    return {"loss": loss}
```

**训练特点**：
- 编码器冻结（DINOv2预训练）
- 只训练瓶颈层和解码器
- 使用余弦硬挖掘损失，防止过拟合

### 4.2 推理流程

```python
# 1. 加载模型
model = Dinomaly.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# 2. 前向传播
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

**推理步骤**：
1. DINOv2编码器提取中间层特征
2. 瓶颈层压缩特征
3. 解码器重建特征
4. 计算编码器和解码器特征的余弦不相似度
5. 融合多尺度异常图
6. 高斯模糊平滑
7. 计算图像级异常分数（最大值或Top-K平均）

## 五、关键实现细节

### 5.1 特征处理

**代码来源**：`anomalib/models/image/dinomaly/torch_model.py` (第377-407行)

```python
def _process_features_for_spatial_output(
    self,
    features: list[torch.Tensor],
    h_patches: int,
    w_patches: int,
) -> list[torch.Tensor]:
    """处理特征用于空间输出
    
    移除class token和register tokens，重塑为空间特征图。
    
    Args:
        features: 特征列表，每个元素是 [B, N_tokens, D]
        h_patches: patch数量（高度）
        w_patches: patch数量（宽度）
    
    Returns:
        处理后的特征列表，每个元素是 [B, D, H, W]
    """
    processed = []
    for feat in features:
        # 移除class token（如果有）
        if self.remove_class_token:
            feat = feat[:, 1:, :]  # [B, N_patches, D]
        
        # 移除register tokens（如果有）
        if hasattr(self.encoder, "num_register_tokens") and self.encoder.num_register_tokens > 0:
            feat = feat[:, :-self.encoder.num_register_tokens, :]
        
        # 重塑为空间特征图
        feat = feat.permute(0, 2, 1).reshape(
            feat.shape[0], 
            feat.shape[2], 
            h_patches, 
            w_patches
        )  # [B, D, H, W]
        
        processed.append(feat)
    
    return processed
```

### 5.2 线性注意力

Dinomaly使用LinearAttention简化计算：
- 降低计算复杂度
- 保持注意力机制的有效性
- 符合"少即是多"的哲学

## 六、算法优缺点分析

### 6.1 优点

1. **简单有效**：编码器-瓶颈-解码器架构，实现简单
2. **DINOv2特征**：使用强大的DINOv2特征，无需训练编码器
3. **硬挖掘损失**：防止解码器过拟合，提升泛化能力
4. **多尺度融合**：融合多个特征组，保留细节信息

### 6.2 缺点

1. **编码器依赖**：依赖DINOv2的质量
2. **训练复杂**：需要训练瓶颈层和解码器
3. **计算量大**：Vision Transformer的计算量较大
4. **内存占用**：需要存储中间层特征

## 七、关键超参数

### 7.1 模型超参数

- **encoder_name**：DINOv2编码器名称（如"dinov2reg_vit_base_14"）
- **decoder_depth**：解码器层数（默认8）
- **bottleneck_dropout**：瓶颈层dropout率（默认0.2）
- **target_layers**：提取特征的编码器层（默认[2,3,4,5,6,7,8,9]）
- **fuse_layer_encoder**：编码器特征融合组（默认[[0,1,2,3], [4,5,6,7]]）
- **fuse_layer_decoder**：解码器特征融合组（默认[[0,1,2,3], [4,5,6,7]]）

### 7.2 损失函数超参数

- **p_final**：最终easy points比例（默认0.9）
- **p_schedule_steps**：p值调度步数（默认1000）
- **factor**：梯度降低因子（默认0.1）

### 7.3 后处理超参数

- **DEFAULT_RESIZE_SIZE**：异常图调整尺寸（默认256）
- **DEFAULT_GAUSSIAN_SIGMA**：高斯模糊标准差（默认4）
- **DEFAULT_MAX_RATIO**：Top-K比例（默认0.01，即1%）

## 八、使用示例

### 8.1 基本使用

```python
from anomalib.models import Dinomaly
from anomalib.data import MVTecAD
from anomalib.engine import Engine

# 初始化模型
model = Dinomaly(
    encoder_name="dinov2reg_vit_base_14",
    decoder_depth=8,
    bottleneck_dropout=0.2
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

Dinomaly是一种基于Vision Transformer的异常检测方法，核心思想是"少即是多"：
1. **DINOv2编码器**：使用预训练的DINOv2提取通用特征
2. **瓶颈层**：MLP压缩特征，学习紧凑表示
3. **解码器**：Vision Transformer块重建特征
4. **余弦硬挖掘损失**：降低easy points的梯度，防止过拟合
5. **余弦相似度**：通过编码器和解码器特征的相似度检测异常

该方法在多个数据集上表现优异，是异常检测的最新方法之一。

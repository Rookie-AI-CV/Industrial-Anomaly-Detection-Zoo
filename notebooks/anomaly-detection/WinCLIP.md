# WinCLIP - Zero-Few-Shot Anomaly Classification and Segmentation 完整算法解读

## 论文信息
- **标题**: WinCLIP: Zero-Few-Shot Anomaly Classification and Segmentation
- **年份**: 2023
- **论文链接**: https://arxiv.org/abs/2303.14814
- **核心思想**: 利用CLIP的视觉-语言对齐能力，通过文本提示和图像特征匹配，实现零样本和少样本的异常分类和分割

## 算法概述

WinCLIP是一种基于CLIP的零样本/少样本异常检测方法。它使用CLIP的视觉-语言对齐能力，通过文本提示描述异常，然后计算图像特征与文本特征的相似度来检测异常。核心创新是使用滑动窗口（sliding window）进行多尺度检测，以及使用参考图像进行少样本学习。

### 整体架构

```
输入图像 → CLIP视觉编码器 → 图像特征（窗口+patch）
文本提示 → CLIP文本编码器 → 文本特征
参考图像 → CLIP视觉编码器 → 参考特征（few-shot）
         ↓
    特征匹配（余弦相似度） → 异常分数 → 异常图
```

## 一、模型架构

### 1.1 CLIP模型初始化

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第59-138行)

WinCLIP使用OpenCLIP作为基础模型，支持零样本和少样本两种模式。

```python
class WinClipModel(DynamicBufferMixin, BufferListMixin, nn.Module):
    """WinCLIP模型：基于CLIP的零样本/少样本异常检测"""
    
    def __init__(
        self,
        class_name: str | None = None,           # 类别名称（用于文本提示）
        reference_images: torch.Tensor | None = None,  # 参考图像（few-shot）
        scales: tuple = (2, 3),                  # 滑动窗口尺度
        apply_transform: bool = False,           # 是否应用CLIP变换
    ) -> None:
        super().__init__()
        
        # CLIP配置
        self.backbone = "ViT-B-16-plus-240"      # CLIP骨干网络
        self.pretrained = "laion400m_e31"        # 预训练权重
        self.temperature = 0.07                   # 温度参数
        
        self.class_name = class_name
        self.reference_images = reference_images
        self.scales = scales
        self.apply_transform = apply_transform
        self.k_shot = 0  # 参考图像数量
        
        # 初始化CLIP模型
        self.clip, _, self._transform = open_clip.create_model_and_transforms(
            self.backbone, 
            pretrained=self.pretrained
        )
        self.clip.visual.output_tokens = True  # 输出patch tokens
        self.grid_size = self.clip.visual.grid_size  # 特征图尺寸（如14x14）
        
        # 注册缓冲区
        self.register_buffer_list("masks", self._generate_masks(), persistent=False)
        self.register_buffer("_text_embeddings", torch.empty(0))
        self.register_buffer_list("_visual_embeddings", [torch.empty(0) for _ in self.scales])
        self.register_buffer("_patch_embeddings", torch.empty(0))
        
        # 设置
        self.setup(class_name, reference_images)
```

**关键配置**：
- **backbone**：使用ViT-B-16-plus-240作为视觉编码器
- **pretrained**：使用laion400m_e31预训练权重
- **output_tokens**：设置为True以获取patch级别的特征
- **grid_size**：特征图尺寸（如14x14，对应196个patches）

### 1.2 文本提示生成

**代码来源**：`anomalib/models/image/winclip/prompting.py`

WinCLIP使用提示集合（prompt ensemble）生成文本特征。

```python
def create_prompt_ensemble(class_name: str | None = None) -> list[str]:
    """创建提示集合
    
    Args:
        class_name: 类别名称（如"transistor", "bottle"）
    
    Returns:
        提示列表
    """
    # 基础提示模板
    templates = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of a {}",
        "a bright photo of a {}",
        "a dark photo of a {}",
    ]
    
    # 异常描述
    anomaly_descriptions = [
        "defect",
        "flaw",
        "scratch",
        "crack",
        "stain",
        "hole",
        "dent",
        "tear",
        "contamination",
        "anomaly",
    ]
    
    prompts = []
    
    # 1. 正常样本提示
    if class_name:
        for template in templates:
            prompts.append(template.format(class_name))
    
    # 2. 异常样本提示
    for template in templates:
        for desc in anomaly_descriptions:
            if class_name:
                prompts.append(template.format(f"{class_name} with {desc}"))
            else:
                prompts.append(template.format(desc))
    
    return prompts
```

**提示设计说明**：
- **正常提示**：描述正常样本（如"a photo of a transistor"）
- **异常提示**：描述异常样本（如"a photo of a transistor with defect"）
- **模板多样性**：使用多个模板增加鲁棒性

### 1.3 文本嵌入收集

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第140-195行)

```python
def _collect_text_embeddings(self, class_name: str) -> None:
    """收集文本嵌入
    
    使用提示集合生成文本特征。
    
    Args:
        class_name: 类别名称
    """
    # 生成提示集合
    prompts = create_prompt_ensemble(class_name)
    
    # Tokenize
    text_tokens = tokenize(prompts).to(self.clip.device)
    
    # 文本编码
    with torch.no_grad():
        text_features = self.clip.encode_text(text_tokens)  # [N_prompts, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # L2归一化
    
    self._text_embeddings = text_features
```

**文本嵌入说明**：
- **提示集合**：使用多个提示描述正常和异常
- **L2归一化**：归一化文本特征，便于计算余弦相似度
- **零样本模式**：只使用文本提示，无需参考图像

### 1.4 参考图像嵌入收集（Few-shot）

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第393-420行)

```python
def _collect_visual_embeddings(self, reference_images: torch.Tensor) -> None:
    """收集参考图像的视觉嵌入（few-shot模式）
    
    Args:
        reference_images: [K, C, H, W] - K个参考正常图像
    """
    self.k_shot = reference_images.shape[0]
    
    # 编码参考图像
    image_embeddings, window_embeddings, patch_embeddings = self.encode_image(reference_images)
    
    # 存储多尺度窗口嵌入
    self._visual_embeddings = window_embeddings
    
    # 存储patch嵌入
    self._patch_embeddings = patch_embeddings
```

## 二、图像编码

### 2.1 图像编码（图像、窗口、patch）

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第197-257行)

WinCLIP提取三种类型的特征：图像级特征、窗口特征和patch特征。

```python
def encode_image(
    self, 
    batch: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """编码图像：获取图像、窗口和patch嵌入
    
    Args:
        batch: [B, C, H, W] - 输入图像
    
    Returns:
        (image_embeddings, window_embeddings, patch_embeddings)
        - image_embeddings: [B, D] - 图像级特征
        - window_embeddings: 窗口特征列表，每个元素是 [B, N_windows, D]
        - patch_embeddings: [B, N_patches, D] - patch特征
    """
    # 应用变换（如果需要）
    if self.apply_transform:
        batch = torch.stack([self.transform(image) for image in batch])
    
    # 注册hook获取中间特征图
    outputs = {}
    
    def get_feature_map(name: str) -> Callable:
        def hook(_model, inputs, _outputs):
            outputs[name] = inputs[0].detach()
        return hook
    
    # 注册hook获取transformer的中间tokens
    self.clip.visual.patch_dropout.register_forward_hook(
        get_feature_map("feature_map")
    )
    
    # 获取图像和patch嵌入
    image_embeddings, patch_embeddings = self.clip.encode_image(batch)
    
    # 获取窗口嵌入（使用滑动窗口）
    feature_map = outputs["feature_map"]  # [B, N_patches, D]
    window_embeddings = [
        self._get_window_embeddings(feature_map, masks) 
        for masks in self.masks
    ]
    
    return image_embeddings, window_embeddings, patch_embeddings
```

### 2.2 窗口嵌入计算

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第259-294行)

```python
def _get_window_embeddings(
    self, 
    feature_map: torch.Tensor,  # [B, N_patches, D]
    masks: torch.Tensor         # [kernel_size, n_masks]
) -> torch.Tensor:
    """计算每个滑动窗口的嵌入
    
    Args:
        feature_map: [B, N_patches, D] - 特征图
        masks: [kernel_size, n_masks] - 窗口位置掩码
    
    Returns:
        [B, n_masks, D] - 每个窗口的嵌入
    """
    batch_size = feature_map.shape[0]
    n_masks = masks.shape[1]
    
    # 添加class token索引（索引0）
    class_index = torch.zeros(1, n_masks, dtype=int).to(feature_map.device)
    masks = torch.cat((class_index, masks + 1)).T  # +1因为class token在索引0
    
    # 应用掩码选择特征
    masked = torch.cat([
        torch.index_select(feature_map, 1, mask) 
        for mask in masks
    ])  # [n_masks * B, kernel_size, D]
    
    # 完成前向传播
    masked = self.clip.visual.patch_dropout(masked)
    masked = self.clip.visual.ln_pre(masked)
    
    masked = masked.permute(1, 0, 2)  # NLD -> LND
    masked = self.clip.visual.transformer(masked)
    masked = masked.permute(1, 0, 2)  # LND -> NLD
    
    masked = self.clip.visual.ln_post(masked)
    pooled, _ = self.clip.visual._global_pool(masked)
    
    if self.clip.visual.proj is not None:
        pooled = pooled @ self.clip.visual.proj
    
    # 重塑为 [B, n_masks, D]
    return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)
```

**窗口嵌入说明**：
- **滑动窗口**：使用不同尺度的滑动窗口（如2x2, 3x3）
- **掩码选择**：每个窗口位置对应一个掩码，选择覆盖的patches
- **特征聚合**：通过transformer和全局池化聚合窗口特征

### 2.3 滑动窗口掩码生成

**代码来源**：`anomalib/models/image/winclip/utils.py` (第246-307行)

```python
def make_masks(
    grid_size: tuple[int, int],  # (H, W)
    kernel_size: int,             # 窗口大小（patches数）
    stride: int = 1               # 步长
) -> torch.Tensor:
    """生成滑动窗口掩码
    
    Args:
        grid_size: 特征图尺寸 (H, W)
        kernel_size: 窗口大小（patches数）
        stride: 步长
    
    Returns:
        [n_patches_per_mask, n_masks] - 每个列表示一个窗口位置的掩码
    """
    height, width = grid_size
    grid = torch.arange(height * width).reshape(1, height, width)
    
    # 使用unfold生成滑动窗口
    return nn.functional.unfold(
        grid.float(), 
        kernel_size=kernel_size, 
        stride=stride
    ).int()
```

## 三、异常分数计算

### 3.1 类别分数计算

**代码来源**：`anomalib/models/image/winclip/utils.py` (第96-166行)

```python
def class_scores(
    image_embeddings: torch.Tensor,  # [B, N, D] 或 [N, D]
    text_embeddings: torch.Tensor,   # [M, D] 或 [B, M, D]
    temperature: float = 1.0,
    target_class: int | None = None,  # 目标类别索引（1表示异常）
) -> torch.Tensor:
    """计算类别分数：图像嵌入与文本嵌入的相似度
    
    使用余弦相似度 + 温度缩放 + softmax
    
    Args:
        image_embeddings: 图像嵌入
        text_embeddings: 文本嵌入
        temperature: 温度参数（默认0.07）
        target_class: 目标类别（1表示异常类）
    
    Returns:
        类别分数
    """
    # 计算余弦相似度
    scores = cosine_similarity(image_embeddings, text_embeddings)
    
    # 温度缩放 + softmax
    scores = (scores / temperature).softmax(dim=-1)
    
    # 如果指定了目标类别，只返回该类别的分数
    if target_class is not None:
        return scores[..., target_class]
    
    return scores
```

### 3.2 零样本异常分数

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第329-356行)

```python
def _compute_zero_shot_scores(
    self,
    image_scores: torch.Tensor,        # [B] - 图像级分数
    window_embeddings: list[torch.Tensor],  # 窗口嵌入列表
) -> torch.Tensor:
    """计算零样本多尺度异常分数图
    
    Args:
        image_scores: [B] - 图像级异常分数
        window_embeddings: 窗口嵌入列表
    
    Returns:
        [B, H, W] - 零样本异常分数图
    """
    # 图像级分数扩展到全图（作为第0个尺度）
    multi_scale_scores = [
        image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])
    ]
    
    # 对每个尺度计算窗口分数
    for window_embedding, mask in zip(window_embeddings, self.masks, strict=True):
        # 计算窗口与文本嵌入的相似度
        scores = class_scores(
            window_embedding, 
            self.text_embeddings, 
            self.temperature, 
            target_class=1  # 异常类
        )  # [B, n_windows]
        
        # 使用调和平均聚合窗口分数到patch位置
        multi_scale_scores.append(
            harmonic_aggregation(scores, self.grid_size, mask)
        )
    
    # 跨尺度聚合（调和平均）
    return (len(self.scales) + 1) / (1 / torch.stack(multi_scale_scores)).sum(dim=0)
```

### 3.3 少样本异常分数

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第358-391行)

```python
def _compute_few_shot_scores(
    self,
    patch_embeddings: torch.Tensor,    # [B, N_patches, D]
    window_embeddings: list[torch.Tensor],  # 窗口嵌入列表
) -> torch.Tensor:
    """计算少样本多尺度异常分数图
    
    使用视觉关联分数（visual association score）计算测试图像与参考图像的差异。
    
    Args:
        patch_embeddings: patch嵌入
        window_embeddings: 窗口嵌入列表
    
    Returns:
        [B, H, W] - 少样本异常分数图
    """
    # 全尺度patch分数
    multi_scale_scores = [
        visual_association_score(
            patch_embeddings, 
            self.patch_embeddings
        ).reshape((-1, *self.grid_size))
    ]
    
    # 对每个尺度计算窗口分数
    for window_embedding, reference_embedding, mask in zip(
        window_embeddings,
        self.visual_embeddings,
        self.masks,
        strict=True,
    ):
        # 计算窗口与参考窗口的视觉关联分数
        scores = visual_association_score(window_embedding, reference_embedding)
        
        # 使用调和平均聚合
        multi_scale_scores.append(
            harmonic_aggregation(scores, self.grid_size, mask)
        )
    
    # 跨尺度平均
    return torch.stack(multi_scale_scores).mean(dim=0)
```

### 3.4 视觉关联分数

**代码来源**：`anomalib/models/image/winclip/utils.py` (第215-243行)

```python
def visual_association_score(
    embeddings: torch.Tensor,              # [B, N, D]
    reference_embeddings: torch.Tensor      # [K, N, D] 或 [K*N, D]
) -> torch.Tensor:
    """计算视觉关联分数
    
    使用最小余弦距离：每个嵌入到参考嵌入的最小距离。
    
    Args:
        embeddings: 测试嵌入
        reference_embeddings: 参考嵌入
    
    Returns:
        [B, N] - 视觉关联分数（值越大表示差异越大）
    """
    # 重塑参考嵌入
    reference_embeddings = reference_embeddings.reshape(-1, embeddings.shape[-1])
    
    # 计算余弦相似度
    scores = cosine_similarity(embeddings, reference_embeddings)  # [B, N, K*N]
    
    # 最小余弦距离（转换为距离）
    return (1 - scores).min(dim=-1)[0] / 2
```

### 3.5 调和平均聚合

**代码来源**：`anomalib/models/image/winclip/utils.py` (第169-212行)

```python
def harmonic_aggregation(
    window_scores: torch.Tensor,  # [B, n_masks] - 窗口分数
    output_size: tuple,            # (H, W)
    masks: torch.Tensor           # [n_patches_per_mask, n_masks]
) -> torch.Tensor:
    """使用调和平均聚合窗口分数到patch位置
    
    调和平均对低分数更敏感，适合异常检测。
    
    Args:
        window_scores: 窗口分数
        output_size: 输出尺寸
        masks: 窗口掩码
    
    Returns:
        [B, H, W] - 聚合后的分数图
    """
    batch_size = window_scores.shape[0]
    height, width = output_size
    
    scores = []
    for idx in range(height * width):
        # 找到包含该patch的所有窗口
        patch_mask = torch.any(masks == idx, dim=0)  # [n_masks]
        
        # 调和平均：n / sum(1/scores)
        scores.append(
            sum(patch_mask) / (1 / window_scores.T[patch_mask]).sum(dim=0)
        )
    
    return torch.stack(scores).T.reshape(batch_size, height, width).nan_to_num(posinf=0.0)
```

## 四、前向传播

### 4.1 完整前向传播流程

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第296-327行)

```python
@torch.no_grad
def forward(self, batch: torch.Tensor) -> InferenceBatch:
    """前向传播：计算图像和像素异常分数
    
    Args:
        batch: [B, C, H, W] - 输入图像
    
    Returns:
        InferenceBatch(pred_score, anomaly_map)
    """
    # 1. 编码图像
    image_embeddings, window_embeddings, patch_embeddings = self.encode_image(batch)
    
    # 2. 零样本分数
    image_scores = class_scores(
        image_embeddings, 
        self.text_embeddings, 
        self.temperature, 
        target_class=1
    )  # [B]
    
    multi_scale_scores = self._compute_zero_shot_scores(image_scores, window_embeddings)
    
    # 3. 少样本分数（如果有参考图像）
    if self.k_shot:
        few_shot_scores = self._compute_few_shot_scores(patch_embeddings, window_embeddings)
        # 融合零样本和少样本分数
        multi_scale_scores = (multi_scale_scores + few_shot_scores) / 2
        image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2
    
    # 4. 上采样到原始图像尺寸
    pixel_scores = nn.functional.interpolate(
        multi_scale_scores.unsqueeze(1),
        size=batch.shape[-2:],
        mode="bilinear",
    )
    
    return InferenceBatch(
        pred_score=image_scores, 
        anomaly_map=pixel_scores.squeeze(1)
    )
```

**前向传播说明**：
- **零样本模式**：只使用文本提示计算异常分数
- **少样本模式**：结合文本提示和参考图像计算异常分数
- **多尺度融合**：融合不同尺度的窗口分数
- **分数融合**：零样本和少样本分数平均融合
## 五、关键实现细节

### 5.1 滑动窗口掩码生成

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第422-440行)

```python
def _generate_masks(self) -> list[torch.Tensor]:
    """生成滑动窗口掩码
    
    Returns:
        掩码列表，每个元素对应一个尺度
    """
    masks = []
    for scale in self.scales:
        mask = make_masks(self.grid_size, kernel_size=scale, stride=1)
        masks.append(mask)
    return masks
```

### 5.2 温度参数

WinCLIP使用温度参数`temperature=0.07`（来自CLIP论文）：
- 温度缩放：`scores / temperature`
- Softmax：将相似度转换为概率分布
- 较低温度使分布更尖锐，突出最相似的提示

### 5.3 调和平均 vs 算术平均

WinCLIP使用调和平均聚合窗口分数：
- **调和平均**：`n / sum(1/scores)`，对低分数更敏感
- **算术平均**：`mean(scores)`，对所有分数同等对待
- 调和平均更适合异常检测，因为低分数（异常）更重要

## 六、完整训练与推理流程

### 6.1 零样本模式

```python
# 1. 初始化模型（只使用类别名称）
model = WinClipModel(class_name="transistor")

# 2. 推理
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score  # 图像级分数
anomaly_map = predictions.anomaly_map    # 像素级异常图
```

### 6.2 少样本模式

```python
# 1. 初始化模型（使用类别名称和参考图像）
reference_images = torch.stack([normal_img1, normal_img2, normal_img3])  # [3, C, H, W]
model = WinClipModel(
    class_name="transistor",
    reference_images=reference_images
)

# 2. 推理（自动融合零样本和少样本分数）
with torch.no_grad():
    predictions = model(test_image)

# 3. 获取结果
anomaly_score = predictions.pred_score
anomaly_map = predictions.anomaly_map
```

## 七、算法优缺点分析

### 7.1 优点

1. **零样本能力**：无需训练，直接使用CLIP的视觉-语言对齐
2. **少样本学习**：使用少量参考图像提升性能
3. **多尺度检测**：使用滑动窗口进行多尺度检测
4. **无需异常样本**：只使用正常样本和文本提示
5. **通用性强**：适用于多种异常类型和领域

### 7.2 缺点

1. **CLIP依赖**：依赖CLIP的预训练质量
2. **计算量大**：滑动窗口和多次前向传播
3. **提示工程**：需要设计合适的文本提示
4. **定位精度**：patch级别的定位可能不够精细

## 八、关键超参数

### 8.1 模型超参数

- **backbone**：CLIP骨干网络（默认"ViT-B-16-plus-240"）
- **pretrained**：预训练权重（默认"laion400m_e31"）
- **scales**：滑动窗口尺度（默认(2, 3)）
- **temperature**：温度参数（默认0.07）
- **class_name**：类别名称（用于文本提示）

### 8.2 提示设计

- **模板数量**：5个基础模板
- **异常描述**：10种异常类型描述
- **提示总数**：约50-60个提示（正常+异常）

## 九、使用示例

### 9.1 基本使用

```python
from anomalib.models import WinClip
from anomalib.data import MVTecAD
from anomalib.engine import Engine

# 零样本模式
model = WinClip(class_name="bottle")

# 加载数据
datamodule = MVTecAD(category="bottle")

# 推理（无需训练）
engine = Engine()
predictions = engine.predict(model=model, datamodule=datamodule)
```

### 9.2 少样本模式

```python
# 少样本模式（使用参考图像）
model = WinClip(
    class_name="bottle",
    reference_images=normal_images  # [K, C, H, W]
)

# 推理
predictions = engine.predict(model=model, datamodule=datamodule)
```

## 十、论文与代码对应关系

### 10.1 论文核心概念

**论文Section 3.1 - Zero-Shot Anomaly Detection**：
- 使用文本提示描述异常
- 计算图像特征与文本特征的相似度

**代码实现**：
- 文本嵌入：`_collect_text_embeddings()` (第393-420行)
- 类别分数：`class_scores()` (第96-166行)
- 零样本分数：`_compute_zero_shot_scores()` (第329-356行)

**论文Section 3.2 - Few-Shot Anomaly Detection**：
- 使用参考图像进行少样本学习
- 计算视觉关联分数

**代码实现**：
- 视觉嵌入：`_collect_visual_embeddings()` (第393-420行)
- 视觉关联分数：`visual_association_score()` (第215-243行)
- 少样本分数：`_compute_few_shot_scores()` (第358-391行)

**论文Section 3.3 - Multi-Scale Detection**：
- 使用滑动窗口进行多尺度检测
- 调和平均聚合窗口分数

**代码实现**：
- 窗口嵌入：`_get_window_embeddings()` (第259-294行)
- 调和聚合：`harmonic_aggregation()` (第169-212行)

### 10.2 关键设计

1. **滑动窗口**：使用不同尺度的滑动窗口（如2x2, 3x3）进行多尺度检测
2. **调和平均**：使用调和平均聚合窗口分数，对低分数更敏感
3. **提示集合**：使用多个提示模板和异常描述，增加鲁棒性
4. **分数融合**：零样本和少样本分数平均融合

## 十一、总结

WinCLIP是一种基于CLIP的零样本/少样本异常检测方法，核心思想是：
1. **文本提示**：使用文本提示描述正常和异常样本
2. **视觉-语言对齐**：利用CLIP的视觉-语言对齐能力
3. **滑动窗口**：使用滑动窗口进行多尺度检测
4. **调和平均**：使用调和平均聚合窗口分数
5. **少样本学习**：使用参考图像进行少样本学习

该方法无需训练即可使用，适用于多种异常类型和领域，是零样本异常检测的代表性方法。

```
1. 加载预训练CLIP模型
2. 准备少量异常和正常样本
3. 可选：微调提示或特征编码器
4. 设计文本提示
5. 提取特征并匹配
6. 进行异常检测
```

### 9. 测试流程总结（零样本）

```
1. 加载预训练CLIP模型
2. 设计文本提示（异常和正常）
3. 对测试图像：
   - 提取视觉特征
   - 提取文本特征
   - 计算相似度
   - 生成异常分数和分割图
4. 应用阈值进行异常判断
```

### 10. 优缺点分析

**优点**：
- 无需训练即可使用（零样本）
- 利用大规模预训练知识
- 可以处理新类别
- 文本提示灵活可定制

**缺点**：
- 依赖文本提示的质量
- 对领域差异敏感
- 可能不如专门训练的方法
- 计算开销较大（需要提取文本特征）

### 11. 关键超参数

- **CLIP模型版本**：ViT-B/32, ViT-L/14等
- **提示模板**：文本提示的设计
- **温度参数**：相似度缩放温度
- **少样本学习率**：微调时的学习率
- **多尺度策略**：使用的尺度列表

## 代码实现详解（基于Anomalib）

### 12. 核心代码结构

基于Anomalib的实现，WinCLIP的核心代码位于 `anomalib/models/image/winclip/torch_model.py`。

#### 12.1 模型初始化

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第59-138行)

```python
class WinClipModel(DynamicBufferMixin, BufferListMixin, nn.Module):
    """WinCLIP模型：基于CLIP的零样本/少样本异常检测"""
    def __init__(
        self,
        class_name: str | None = None,
        reference_images: torch.Tensor | None = None,
        scales: tuple = (2, 3),
    ) -> None:
        # 加载CLIP模型
        self.clip, _, self._transform = open_clip.create_model_and_transforms(
            "ViT-B-16-plus-240", 
            pretrained="laion400m_e31"
        )
        self.clip.visual.output_tokens = True  # 输出patch tokens
        
        # 生成滑动窗口掩码
        self.masks = self._generate_masks()
        
        # 注册buffer存储嵌入
        self.register_buffer("_text_embeddings", torch.empty(0))
        self.register_buffer_list("_visual_embeddings", [...])
        
        # 设置
        self.setup(class_name, reference_images)
```

#### 12.2 文本嵌入收集

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第393-426行)

```python
def _collect_text_embeddings(self, class_name: str) -> None:
    """收集文本嵌入（提示集成）"""
    # 创建提示集成
    normal_prompts, anomalous_prompts = create_prompt_ensemble(class_name)
    
    # 编码提示
    normal_tokens = tokenize(normal_prompts)
    anomalous_tokens = tokenize(anomalous_prompts)
    
    normal_embeddings = self.clip.encode_text(normal_tokens)
    anomalous_embeddings = self.clip.encode_text(anomalous_tokens)
    
    # 平均并拼接
    normal_emb = torch.mean(normal_embeddings, dim=0, keepdim=True)
    anomalous_emb = torch.mean(anomalous_embeddings, dim=0, keepdim=True)
    self._text_embeddings = torch.cat((normal_emb, anomalous_emb))
```

#### 12.3 窗口嵌入提取

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第259-294行)

```python
def _get_window_embeddings(
    self, 
    feature_map: torch.Tensor, 
    masks: torch.Tensor
) -> torch.Tensor:
    """提取滑动窗口嵌入"""
    # 应用掩码选择patches
    masked = torch.cat([
        torch.index_select(feature_map, 1, mask) 
        for mask in masks
    ])
    
    # 通过transformer
    masked = self.clip.visual.transformer(masked)
    
    # 全局池化
    pooled, _ = self.clip.visual._global_pool(masked)
    
    return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)
```

#### 12.4 零样本/少样本分数计算

**代码来源**：`anomalib/models/image/winclip/torch_model.py` (第296-327行)

```python
def forward(self, batch: torch.Tensor) -> InferenceBatch:
    """前向传播"""
    # 编码图像
    image_embeddings, window_embeddings, patch_embeddings = self.encode_image(batch)
    
    # 零样本分数
    image_scores = class_scores(
        image_embeddings, 
        self.text_embeddings, 
        self.temperature, 
        target_class=1
    )
    multi_scale_scores = self._compute_zero_shot_scores(image_scores, window_embeddings)
    
    # 少样本分数（如果有参考图像）
    if self.k_shot:
        few_shot_scores = self._compute_few_shot_scores(patch_embeddings, window_embeddings)
        multi_scale_scores = (multi_scale_scores + few_shot_scores) / 2
    
    # 上采样到原始尺寸
    pixel_scores = F.interpolate(
        multi_scale_scores.unsqueeze(1),
        size=batch.shape[-2:],
        mode="bilinear"
    )
    return InferenceBatch(pred_score=image_scores, anomaly_map=pixel_scores.squeeze(1))
```

**关键实现**：
- 使用滑动窗口提取局部特征
- 零样本使用文本嵌入，少样本使用视觉嵌入
- 多尺度融合提升检测性能


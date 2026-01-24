# DiAD - A Diffusion-based Framework for Multi-class Anomaly Detection 实现详解

## 论文信息
- **标题**: DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection
- **年份**: 2024
- **核心思想**: 使用扩散模型学习正常数据的分布，通过去噪过程重建图像，利用重建误差和扩散过程的异常分数进行多类异常检测

**注意**：本文档中的代码示例为伪代码，用于说明算法原理。DiAD在Anomalib中暂无官方实现。

## 整体架构

```
输入图像 → 扩散过程（加噪） → 去噪过程（重建） → 重建误差 → 异常检测
         ↓
    扩散模型（学习正常分布）
```

## 详细实现过程

### 1. 扩散模型基础

#### 1.1 前向扩散过程
- 逐步向图像添加高斯噪声
- 经过T步后，图像变成纯噪声
- q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

```python
# 伪代码
def forward_diffusion(x_0, num_steps, betas):
    """
    前向扩散过程：逐步加噪
    x_0: [B, C, H, W] - 原始图像
    num_steps: 扩散步数
    betas: [num_steps] - 噪声调度
    """
    B, C, H, W = x_0.shape
    
    # 计算累积乘积
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # 随机选择时间步
    t = torch.randint(0, num_steps, (B,))
    
    # 计算噪声
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    
    # 采样噪声
    noise = torch.randn_like(x_0)
    
    # 加噪
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    return x_t, t, noise
```

#### 1.2 反向去噪过程
- 学习从噪声重建图像
- 使用神经网络预测噪声
- p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

```python
# 伪代码
class DiffusionModel(nn.Module):
    """
    扩散模型：学习去噪过程
    """
    def __init__(self, in_channels=3, time_embed_dim=256):
        super().__init__()
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # U-Net去噪网络
        self.unet = UNet(in_channels, in_channels, time_embed_dim)
    
    def forward(self, x_t, t):
        """
        x_t: [B, C, H, W] - 加噪图像
        t: [B] - 时间步
        预测噪声
        """
        # 时间嵌入
        t_embed = sinusoidal_embedding(t, self.time_embed_dim)
        t_embed = self.time_embed(t_embed)
        
        # 预测噪声
        predicted_noise = self.unet(x_t, t_embed)
        
        return predicted_noise

def sinusoidal_embedding(t, dim):
    """
    正弦位置编码
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

### 2. 训练阶段

#### 2.1 损失函数
- 预测噪声与真实噪声的差异
- L = E[||ε - ε_θ(x_t, t)||²]
- 只使用正常样本训练

```python
# 伪代码
def compute_diffusion_loss(model, x_0, betas, num_steps):
    """
    计算扩散损失
    """
    # 前向扩散
    x_t, t, noise = forward_diffusion(x_0, num_steps, betas)
    
    # 预测噪声
    predicted_noise = model(x_t, t)
    
    # 损失：预测噪声与真实噪声的MSE
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss

def train_epoch(diffusion_model, train_loader, optimizer, betas, num_steps):
    """
    训练扩散模型
    """
    diffusion_model.train()
    
    for normal_images in train_loader:
        # 计算损失
        loss = compute_diffusion_loss(diffusion_model, normal_images, betas, num_steps)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 2.2 多类训练
- 为每个类别训练独立的扩散模型
- 或使用条件扩散模型
- 共享部分参数

```python
# 伪代码
class ConditionalDiffusionModel(nn.Module):
    """
    条件扩散模型：支持多类
    """
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        # 类别嵌入
        self.class_embed = nn.Embedding(num_classes, 256)
        
        # 扩散模型
        self.diffusion = DiffusionModel(in_channels)
    
    def forward(self, x_t, t, class_id):
        """
        class_id: [B] - 类别ID
        """
        # 类别嵌入
        class_emb = self.class_embed(class_id)
        
        # 与时间嵌入融合
        t_embed = self.diffusion.time_embed(sinusoidal_embedding(t, 256))
        combined_embed = t_embed + class_emb
        
        # 预测噪声
        predicted_noise = self.diffusion.unet(x_t, combined_embed)
        
        return predicted_noise
```

### 3. 测试阶段

#### 3.1 图像重建
- 从纯噪声开始，逐步去噪
- 使用训练好的扩散模型
- 生成重建图像

```python
# 伪代码
def sample_from_diffusion(model, shape, num_steps, betas, device):
    """
    从扩散模型采样（重建图像）
    """
    B, C, H, W = shape
    
    # 从纯噪声开始
    x_t = torch.randn(shape, device=device)
    
    # 计算alphas
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # 逐步去噪
    for t in range(num_steps - 1, -1, -1):
        # 当前时间步
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        
        # 预测噪声
        predicted_noise = model(x_t, t_tensor)
        
        # 计算去噪后的图像
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        # 去噪步骤
        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - (betas[t] / torch.sqrt(1 - alpha_t_cumprod[t])) * predicted_noise
        )
        
        if t > 0:
            # 添加随机噪声（采样过程）
            noise = torch.randn_like(x_t)
            x_t = x_t + torch.sqrt(betas[t]) * noise
    
    return x_t
```

#### 3.2 异常分数计算
- **重建误差**：原始图像与重建图像的差异
- **扩散异常分数**：去噪过程中的异常指标
- 结合两种分数

```python
# 伪代码
def compute_anomaly_score(model, test_image, num_steps, betas):
    """
    计算异常分数
    """
    # 方法1：重建误差
    reconstructed = sample_from_diffusion(model, test_image.shape, num_steps, betas, test_image.device)
    recon_error = F.mse_loss(test_image, reconstructed, reduction='none')  # [B, C, H, W]
    recon_error = recon_error.mean(dim=1)  # [B, H, W] - 对RGB通道求平均
    
    # 方法2：扩散异常分数
    # 在去噪过程中，异常样本的去噪可能更困难
    diffusion_score = compute_diffusion_anomaly_score(model, test_image, num_steps, betas)
    
    # 结合两种分数
    anomaly_score = recon_error + 0.5 * diffusion_score
    
    return anomaly_score, recon_error, diffusion_score

def compute_diffusion_anomaly_score(model, test_image, num_steps, betas):
    """
    计算扩散过程的异常分数
    """
    # 对测试图像进行前向扩散
    x_t, t, true_noise = forward_diffusion(test_image, num_steps, betas)
    
    # 预测噪声
    predicted_noise = model(x_t, t)
    
    # 预测误差：异常样本的预测误差可能更大
    prediction_error = F.mse_loss(predicted_noise, true_noise, reduction='none')
    prediction_error = prediction_error.mean(dim=[1, 2, 3])  # [B]
    
    return prediction_error.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
```

### 4. 多类异常检测

#### 4.1 类别特定检测
- 对每个类别使用对应的扩散模型
- 计算类别特定的异常分数
- 选择最可能的类别

```python
# 伪代码
def multi_class_anomaly_detection(test_image, class_models, num_steps, betas):
    """
    多类异常检测
    class_models: 每个类别的扩散模型
    """
    num_classes = len(class_models)
    class_scores = []
    
    for class_id in range(num_classes):
        model = class_models[class_id]
        # 计算该类别的异常分数
        score, _, _ = compute_anomaly_score(model, test_image, num_steps, betas)
        class_scores.append(score)
    
    # 选择异常分数最小的类别（最可能是正常）
    class_scores = torch.stack(class_scores)  # [num_classes, B, H, W]
    min_scores, predicted_class = class_scores.min(dim=0)  # [B, H, W]
    
    # 如果最小分数仍然很大，则判定为异常
    anomaly_map = min_scores
    
    return anomaly_map, predicted_class
```

#### 4.2 条件生成检测
- 使用条件扩散模型
- 对每个类别条件生成
- 比较不同类别的重建质量

```python
# 伪代码
def conditional_multi_class_detection(test_image, conditional_model, num_classes, num_steps, betas):
    """
    使用条件扩散模型进行多类检测
    """
    class_scores = []
    
    for class_id in range(num_classes):
        class_tensor = torch.full((test_image.shape[0],), class_id, device=test_image.device)
        
        # 条件重建
        reconstructed = conditional_sample(
            conditional_model, test_image.shape, num_steps, betas,
            class_tensor, test_image.device
        )
        
        # 计算重建误差
        recon_error = F.mse_loss(test_image, reconstructed, reduction='none').mean(dim=1)
        class_scores.append(recon_error)
    
    class_scores = torch.stack(class_scores)
    min_scores, predicted_class = class_scores.min(dim=0)
    
    return min_scores, predicted_class
```

### 5. 关键实现细节

#### 5.1 噪声调度
- 设计合适的噪声调度（betas）
- 线性调度：β_t = t/T * β_max
- 余弦调度：更平滑的过渡

```python
# 伪代码
def get_noise_schedule(num_steps, schedule_type='linear', beta_start=0.0001, beta_end=0.02):
    """
    生成噪声调度
    """
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_steps)
    elif schedule_type == 'cosine':
        # 余弦调度
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
    
    return betas
```

#### 5.2 采样加速
- 使用DDIM（Denoising Diffusion Implicit Models）加速采样
- 减少采样步数
- 保持重建质量

```python
# 伪代码
def ddim_sample(model, shape, num_steps, betas, eta=0.0, device='cuda'):
    """
    DDIM采样：加速采样过程
    eta=0: DDIM, eta=1: DDPM
    """
    # 简化版本
    x_t = torch.randn(shape, device=device)
    
    # 使用更少的步数
    step_size = num_steps // 50  # 例如，只用50步
    
    for i in range(step_size - 1, -1, -1):
        t = i * (num_steps // step_size)
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        predicted_noise = model(x_t, t_tensor)
        
        # DDIM更新
        # ...（简化实现）
    
    return x_t
```

#### 5.3 特征空间扩散
- 可以在特征空间进行扩散
- 减少计算量
- 提升效率

```python
# 伪代码
class FeatureSpaceDiffusion(nn.Module):
    """
    特征空间扩散模型
    """
    def __init__(self, feature_extractor, feature_dim=256):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.diffusion = DiffusionModel(feature_dim, feature_dim)
    
    def forward(self, x, t):
        """
        在特征空间进行扩散
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 在特征空间扩散
        predicted_noise = self.diffusion(features, t)
        
        return predicted_noise
```

### 6. 训练流程总结

```
1. 初始化扩散模型（U-Net）
2. 设计噪声调度（betas）
3. 对每个训练epoch：
   a. 对正常训练图像：
      - 前向扩散（加噪）
      - 预测噪声
      - 计算损失
      - 反向传播
4. 保存训练好的扩散模型
```

### 7. 测试流程总结

```
1. 加载训练好的扩散模型
2. 对测试图像：
   - 方法1：从噪声重建，计算重建误差
   - 方法2：计算扩散过程的异常分数
   - 结合两种分数
   - 生成异常图
3. 多类检测：对每个类别计算分数，选择最佳类别
4. 应用阈值进行异常判断
```

### 8. 优缺点分析

**优点**：
- 扩散模型强大的生成能力
- 可以学习复杂的正常数据分布
- 重建误差提供直观的异常指示
- 支持多类异常检测

**缺点**：
- 训练和推理时间较长
- 需要大量计算资源
- 采样过程较慢（即使使用DDIM）
- 模型参数较多

### 9. 关键超参数

- **扩散步数**：num_steps（如1000）
- **噪声调度**：betas的生成方式
- **U-Net结构**：去噪网络的层数和通道数
- **采样方法**：DDPM或DDIM
- **学习率**：训练学习率


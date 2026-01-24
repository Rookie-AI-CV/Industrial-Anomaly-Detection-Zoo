# SoftPatch - Unsupervised Anomaly Detection with Noisy Data 实现详解

## 论文信息
- **标题**: SoftPatch: Unsupervised Anomaly Detection with Noisy Data
- **年份**: 2022
- **核心思想**: 针对噪声数据，使用软patch选择机制，通过评估patch的质量和可靠性，动态调整patch权重

**注意**：本文档中的代码示例为伪代码，用于说明算法原理。SoftPatch在Anomalib中暂无官方实现。

## 整体架构

```
输入图像 → Patch提取 → Patch质量评估 → 软权重分配 → 异常检测 → 异常分数
```

## 详细实现过程

### 1. Patch提取阶段

#### 1.1 特征提取
- 使用预训练CNN提取特征
- 从多个中间层提取特征图
- 保持特征图的空间结构

```python
# 伪代码
def extract_features(image, backbone):
    """
    提取多尺度特征
    """
    x = backbone.conv1(image)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    
    x = backbone.layer1(x)
    f1 = backbone.layer2(x)  # [B, 512, H/8, W/8]
    f2 = backbone.layer3(f1)  # [B, 1024, H/16, W/16]
    
    return [f1, f2]
```

#### 1.2 Patch特征提取
- 将特征图划分为patch
- 每个patch对应特征图上的一个空间位置
- 提取每个patch的特征向量

```python
# 伪代码
def extract_patch_features(features):
    """
    提取patch特征
    features: [B, C, H, W]
    """
    B, C, H, W = features.shape
    # 每个空间位置是一个patch
    patch_features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    return patch_features
```

### 2. Patch质量评估

#### 2.1 质量评估指标
- **一致性**：patch特征与周围patch的一致性
- **稳定性**：在不同图像中该patch特征的稳定性
- **代表性**：patch是否代表正常模式

```python
# 伪代码
class PatchQualityEvaluator:
    """
    Patch质量评估器
    """
    def __init__(self):
        self.normal_patch_features = []  # 存储正常patch特征
    
    def evaluate_consistency(self, patch_feat, neighbor_feats):
        """
        评估patch与邻居的一致性
        """
        # 计算与邻居的平均相似度
        similarities = F.cosine_similarity(
            patch_feat.unsqueeze(0),
            neighbor_feats,
            dim=1
        )
        consistency = similarities.mean()
        return consistency
    
    def evaluate_stability(self, patch_feat, historical_feats):
        """
        评估patch的稳定性
        """
        if len(historical_feats) == 0:
            return 1.0
        
        # 计算与历史特征的平均相似度
        similarities = F.cosine_similarity(
            patch_feat.unsqueeze(0),
            historical_feats,
            dim=1
        )
        stability = similarities.mean()
        return stability
    
    def evaluate_quality(self, patch_feat, neighbor_feats, historical_feats):
        """
        综合评估patch质量
        """
        consistency = self.evaluate_consistency(patch_feat, neighbor_feats)
        stability = self.evaluate_stability(patch_feat, historical_feats)
        
        # 综合质量分数
        quality = (consistency + stability) / 2
        return quality
```

#### 2.2 软权重计算
- 根据patch质量计算软权重
- 高质量patch获得高权重
- 低质量（可能是噪声）patch获得低权重

```python
# 伪代码
def compute_soft_weights(patch_features, quality_scores, temperature=1.0):
    """
    计算软权重
    patch_features: [N, C] - patch特征
    quality_scores: [N] - 质量分数
    """
    # 使用softmax将质量分数转换为权重
    # 添加温度参数控制分布的尖锐程度
    weights = F.softmax(quality_scores / temperature, dim=0)
    
    # 或者使用简单的归一化
    # weights = quality_scores / (quality_scores.sum() + 1e-8)
    
    return weights
```

### 3. 训练阶段

#### 3.1 正常Patch特征收集
- 收集所有正常训练图像的patch特征
- 建立正常patch特征库
- 记录每个patch位置的历史特征

```python
# 伪代码
def collect_normal_patches(train_loader, backbone):
    """
    收集正常patch特征
    """
    all_patches = []
    patch_histories = {}  # 每个位置的历史特征
    
    with torch.no_grad():
        for images in train_loader:
            features = extract_features(images, backbone)
            
            for feat in features:
                B, C, H, W = feat.shape
                patch_feat = extract_patch_features(feat)  # [B*H*W, C]
                
                all_patches.append(patch_feat)
                
                # 更新历史特征
                for i in range(H * W):
                    pos_key = i
                    if pos_key not in patch_histories:
                        patch_histories[pos_key] = []
                    patch_histories[pos_key].append(patch_feat[i])
    
    all_patches = torch.cat(all_patches, dim=0)
    return all_patches, patch_histories
```

#### 3.2 质量评估和权重计算
- 对每个patch评估质量
- 计算软权重
- 使用加权特征进行异常检测

```python
# 伪代码
def compute_patch_weights(patch_features, patch_histories, neighbor_size=3):
    """
    计算patch权重
    """
    N, C = patch_features.shape
    H = int(N ** 0.5)  # 假设是方形特征图
    W = H
    
    quality_scores = []
    
    for i in range(N):
        patch_feat = patch_features[i]
        
        # 获取邻居patch
        h_idx, w_idx = i // W, i % W
        neighbor_indices = []
        for dh in range(-neighbor_size//2, neighbor_size//2+1):
            for dw in range(-neighbor_size//2, neighbor_size//2+1):
                nh, nw = h_idx + dh, w_idx + dw
                if 0 <= nh < H and 0 <= nw < W:
                    neighbor_indices.append(nh * W + nw)
        
        neighbor_feats = patch_features[neighbor_indices]
        
        # 获取历史特征
        historical_feats = torch.stack(patch_histories.get(i, []))
        
        # 评估质量
        quality = evaluate_quality(patch_feat, neighbor_feats, historical_feats)
        quality_scores.append(quality)
    
    quality_scores = torch.tensor(quality_scores)
    weights = compute_soft_weights(patch_features, quality_scores)
    
    return weights, quality_scores
```

### 4. 异常检测阶段

#### 4.1 加权异常检测
- 使用软权重调整patch的重要性
- 高质量patch对异常检测贡献更大
- 低质量patch（可能是噪声）的影响被抑制

```python
# 伪代码
def weighted_anomaly_detection(test_features, normal_patches, patch_weights):
    """
    加权异常检测
    """
    # 提取测试patch特征
    test_patches = extract_patch_features(test_features)  # [N, C]
    
    # 计算到正常patch的距离（使用加权）
    # 方法1：加权最近邻
    distances = torch.cdist(test_patches, normal_patches)  # [N_test, N_normal]
    
    # 应用权重到正常patch
    weighted_distances = distances * patch_weights.unsqueeze(0)  # [N_test, N_normal]
    
    # 找到加权最近邻
    nn_distances = weighted_distances.min(dim=1)[0]
    
    # 方法2：加权平均距离
    # weighted_avg_dist = (distances * patch_weights.unsqueeze(0)).mean(dim=1)
    
    return nn_distances
```

#### 4.2 异常分数计算
- 对每个测试patch计算异常分数
- 使用软权重调整分数
- 生成异常分数图

```python
# 伪代码
def compute_anomaly_score_with_weights(test_image, backbone, normal_patches, patch_weights):
    """
    使用软权重计算异常分数
    """
    # 提取特征
    features = extract_features(test_image, backbone)
    
    # 对每个特征层
    anomaly_maps = []
    for feat in features:
        # 提取patch特征
        test_patches = extract_patch_features(feat)
        
        # 加权异常检测
        patch_scores = weighted_anomaly_detection(feat, normal_patches, patch_weights)
        
        # 重塑为空间图
        B, C, H, W = feat.shape
        anomaly_map = patch_scores.reshape(B, H, W)
        anomaly_maps.append(anomaly_map)
    
    # 融合多尺度
    # 上采样到相同尺寸
    target_size = anomaly_maps[0].shape[1:]
    for i in range(1, len(anomaly_maps)):
        anomaly_maps[i] = F.interpolate(
            anomaly_maps[i].unsqueeze(1),
            size=target_size,
            mode='bilinear'
        ).squeeze(1)
    
    final_anomaly_map = torch.stack(anomaly_maps).mean(dim=0)
    
    return final_anomaly_map
```

### 5. 自适应权重更新

#### 5.1 在线权重更新
- 在测试过程中可以动态更新权重
- 根据新观察到的patch特征调整权重
- 适应数据分布的变化

```python
# 伪代码
class AdaptiveWeightUpdater:
    """
    自适应权重更新器
    """
    def __init__(self, initial_weights, update_rate=0.1):
        self.weights = initial_weights
        self.update_rate = update_rate
    
    def update(self, new_patch_features, quality_scores):
        """
        更新权重
        """
        # 计算新权重
        new_weights = compute_soft_weights(new_patch_features, quality_scores)
        
        # 指数移动平均更新
        self.weights = (1 - self.update_rate) * self.weights + \
                       self.update_rate * new_weights
        
        return self.weights
```

### 6. 关键实现细节

#### 6.1 噪声识别
- 通过质量评估识别可能的噪声patch
- 噪声patch通常具有低一致性或低稳定性
- 降低噪声patch的权重

#### 6.2 权重平滑
- 对权重进行平滑处理，避免突变
- 使用空间平滑（邻居patch权重相似）
- 使用时间平滑（历史权重影响当前权重）

```python
# 伪代码
def smooth_weights(weights, spatial_smooth=True, temporal_smooth=True):
    """
    平滑权重
    """
    if spatial_smooth:
        # 空间平滑：使用平均池化
        weights_2d = weights.reshape(H, W)
        weights_2d = F.avg_pool2d(
            weights_2d.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            padding=1
        ).squeeze()
        weights = weights_2d.reshape(-1)
    
    if temporal_smooth:
        # 时间平滑：与历史权重平均
        if hasattr(self, 'historical_weights'):
            weights = 0.7 * weights + 0.3 * self.historical_weights
        self.historical_weights = weights
    
    return weights
```

#### 6.3 阈值自适应
- 根据patch权重动态调整异常检测阈值
- 高质量区域使用更严格的阈值
- 低质量区域使用更宽松的阈值

### 7. 训练流程总结

```
1. 加载预训练特征提取器
2. 收集正常训练图像的patch特征
3. 对每个patch：
   - 评估质量（一致性、稳定性）
   - 计算软权重
4. 保存patch特征库和权重
```

### 8. 测试流程总结

```
1. 加载patch特征库和权重
2. 对测试图像：
   - 提取patch特征
   - 使用加权异常检测
   - 生成异常分数图
3. 可选：动态更新权重
4. 应用阈值进行异常判断
```

### 9. 优缺点分析

**优点**：
- 有效处理噪声数据
- 软权重机制灵活
- 可以适应数据分布变化
- 提升检测的鲁棒性

**缺点**：
- 需要评估patch质量，计算开销较大
- 权重计算需要调优
- 对质量评估指标依赖较大
- 可能误判正常但罕见的patch为噪声

### 10. 关键超参数

- **质量评估指标权重**：一致性和稳定性的权重
- **温度参数**：softmax温度，控制权重分布
- **邻居大小**：用于一致性评估的邻居范围
- **权重更新率**：自适应更新的学习率
- **平滑参数**：空间和时间平滑的强度


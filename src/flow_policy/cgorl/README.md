# C-GoRL 实现说明

## 一、核心改动：用CURL替换"重置+KL对齐N(0,1)"

### 1.1 原始GoRL的问题

```
原始GoRL每个Stage:
┌─────────────────────────────────────────────────────────────┐
│  Stage N 结束: encoder_θ^(N) 学到了有用的状态表征              │
│                     ↓                                        │
│  Stage N+1 开始: encoder_θ^(N+1) = random_init()  ← 完全丢弃! │
│                     ↓                                        │
│  KL正则化: 强制 π_θ(ε|s) ≈ N(0,I)                            │
│                     ↓                                        │
│  问题: 算力浪费 + 训练震荡 + 表征不连续                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 C-GoRL的解决方案

```
C-GoRL每个Stage:
┌─────────────────────────────────────────────────────────────┐
│  Stage N 结束: (curl_ω^(N), policy_θ^(N)) 学到了有用表征      │
│                     ↓                                        │
│  Stage N+1 开始: 继承参数! (curl_ω^(N), policy_θ^(N))         │
│                     ↓                                        │
│  CURL损失: L_InfoNCE 保持表征稳定                             │
│  弱KL正则: λ2 * KL(π_θ || N(0,I))  [可选]                    │
│                     ↓                                        │
│  优势: 保留知识 + 平滑训练 + 表征连续                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、架构对比

### 2.1 原始GoRL架构

```
obs ──► encoder_θ(obs) ──► μ(s), σ(s) ──► ε ~ N(μ(s), σ(s))
                                                │
                                                ▼
obs ─────────────────────────────► decoder_φ(obs, ε) ──► action

关键: encoder直接从原始obs映射到ε分布
稳定性: 依赖KL正则化强制π_θ ≈ N(0,I)
```

### 2.2 C-GoRL架构

```
obs ──► CURL_ω(obs) ──► z_s ──► policy_θ(z_s) ──► ε ~ N(μ(z_s), σ(z_s))
        ~~~~~~~~~~~     ~~~                              │
        对比编码器      表征                              │
                                                         ▼
obs ─────────────────────────────────────► decoder_φ(obs, ε) ──► action

关键: 新增CURL编码器提供稳定表征z_s
稳定性: CURL的InfoNCE损失保证表征一致性
```

### 2.3 信息流对比

| 阶段 | 原始GoRL | C-GoRL |
|-----|---------|--------|
| 观测处理 | obs直接输入encoder | obs → CURL → z_s |
| 策略输入 | obs_norm (归一化观测) | z_s (CURL表征) |
| 策略输出 | ε ~ π_θ(·\|obs) | ε ~ π_θ(·\|z_s) |
| 解码器输入 | (obs, ε) | (obs, ε) [不变!] |
| Stage间继承 | ❌ 重置encoder | ✅ 继承CURL+policy |

---

## 三、核心组件解析

### 3.1 CURL编码器 (`CURLEncoderState`)

```python
class CURLEncoderState:
    query_params: MlpWeights   # f_ω (在线编码器，接收梯度)
    key_params: MlpWeights     # f_ω^EMA (动量编码器，无梯度)
    W: Array                   # 双线性矩阵 (z_dim × z_dim)
```

**功能**：将原始观测映射到稳定的低维表征

**InfoNCE损失**：
```python
def compute_infonce_loss(z_query, z_key, W, temperature):
    # 相似度矩阵: logits[i,j] = z_q[i]^T @ W @ z_k[j]
    logits = jnp.einsum('id,de,je->ij', z_query, W, z_key) / temperature
    
    # 对角线是正样本 (同一观测的两个增强视角)
    labels = jnp.arange(batch_size)
    
    # 交叉熵损失
    loss = softmax_cross_entropy(logits, labels)
    return mean(loss)
```

**EMA更新** (保持key encoder稳定)：
```python
def update_ema(momentum=0.95):
    key_params = momentum * key_params + (1 - momentum) * query_params
```

### 3.2 数据增强 (`augment_state`)

原始CURL使用像素级random crop，但GoRL使用低维状态向量，因此我们采用简单的高斯噪声：

```python
def augment_state(obs, prng, scale=0.01):
    noise = jax.random.normal(prng, obs.shape) * scale
    return obs + noise
```

**为什么这样有效**：
- State-based RL的观测已经是紧凑表征
- 小噪声创造正样本对，同时保持语义
- 避免过拟合到精确的观测值

### 3.3 联合损失函数

```python
def _compute_combined_loss():
    # ========== 1. CURL损失 ==========
    obs_q = augment_state(obs, prng1)
    obs_k = augment_state(obs, prng2)
    z_query = curl_state.encode_query(obs_q)
    z_key = curl_state.encode_key(obs_k)  # stop_gradient!
    curl_loss = compute_infonce_loss(z_query, z_key, W, temperature)
    
    # ========== 2. PPO损失 (在CURL表征空间) ==========
    z_s = curl_state.encode_query(obs_norm)  # CURL表征
    eps_dist = gaussian_policy_fwd(policy_params, z_s)  # 基于z_s的策略
    
    # 标准PPO: 似然比裁剪 + 价值损失 + 熵正则化
    ppo_loss = clipped_surrogate_loss + value_loss + entropy_loss
    
    # ========== 3. KL正则化 (可选) ==========
    # 变体1: λ2 > 0，轻微约束ε分布接近N(0,I)
    # 变体2: λ2 = 0，完全依赖CURL
    kl_loss = kl_coeff * (mean(μ²) + mean(σ²))
    
    # ========== 总损失 ==========
    total_loss = ppo_loss + curl_coeff * curl_loss + kl_loss
```

---

## 四、训练流程对比

### 4.1 原始GoRL训练流程

```
for stage in range(num_stages):
    # ❌ 每个stage重置encoder
    encoder = EncoderState.init(prng)  # 随机初始化
    
    # Phase 1: Encoder训练
    for iteration in range(num_iterations):
        transitions = rollout(encoder, decoder)
        encoder = encoder.training_step(transitions)
        # KL正则化强制 π_θ ≈ N(0,I)
    
    # Phase 2: 收集数据 + Decoder训练
    data = collect_data(encoder, decoder)
    decoder = train_decoder(data)
```

### 4.2 C-GoRL训练流程

```
encoder = None

for stage in range(num_stages):
    # ✅ 继承上一stage的encoder
    if encoder is None:
        encoder = CGoRLEncoderState.init(prng)
    # else: 直接使用上一stage的encoder!
    
    # Phase 1: CURL + PPO联合训练
    for iteration in range(num_iterations):
        transitions = rollout(encoder, decoder)
        encoder = encoder.training_step(transitions)
        # CURL损失保持表征稳定
        # 可选弱KL正则化
    
    # Phase 2: 收集数据 + Decoder训练 (与原GoRL相同)
    data = collect_data(encoder, decoder)
    decoder = train_decoder(data)
```

---

## 五、两个变体

### 5.1 变体1：CURL + 弱KL (推荐)

```python
config = CGoRLConfig(
    curl_coeff=1.0,    # λ1: CURL权重
    kl_coeff=0.001,    # λ2: 弱KL约束
)
```

**优势**：
- CURL提供表征稳定性
- 弱KL防止ε分布漂移太远
- 最平衡的方案

### 5.2 变体2：CURL + 无KL

```python
config = CGoRLConfig(
    curl_coeff=1.0,    # λ1: CURL权重
    kl_coeff=0.0,      # λ2: 无KL约束
)
```

**优势**：
- 完全依赖CURL
- 更少的超参数
- 用于消融实验

**风险**：
- ε分布可能漂移到decoder训练分布之外
- 需要监控ε的均值和方差

---

## 六、文件结构

```
GoRL-main/
├── src/flow_policy/
│   ├── [原有文件 - 完全不修改]
│   │   ├── encoder_ppo.py
│   │   ├── decoder_fm.py
│   │   ├── networks.py
│   │   ├── math_utils.py
│   │   ├── rollouts.py
│   │   └── agent.py
│   │
│   └── cgorl/                  🆕 新建
│       ├── __init__.py         导出接口
│       └── cgorl.py            ~500行，所有C-GoRL组件
│
└── scripts/
    ├── [原有文件 - 完全不修改]
    └── run_cgorl.py            🆕 新建，~400行，训练流水线
```

---

## 七、复用关系

### 7.1 完全复用（无任何修改）

| 原文件 | C-GoRL中如何使用 |
|-------|-----------------|
| `networks.py` | 导入 `mlp_init`, `gaussian_policy_fwd`, `value_mlp_fwd` |
| `math_utils.py` | 导入 `RunningStats`, `NormalDistribution` |
| `decoder_fm.py` | 导入 `DecoderFMState`, `DecoderFMConfig` |
| `rollouts.py` | 导入 `TransitionStruct`, `compute_gae` |

### 7.2 参考但重新实现

| 原文件 | C-GoRL中的对应 | 差异 |
|-------|---------------|------|
| `encoder_ppo.py` | `CGoRLEncoderState` | 新增CURL编码器，输入从obs改为z_s |
| `agent.py` | `CGoRLAgent` | 组合CGoRLEncoderState + DecoderFMState |
| `rollout_encoder.py` | `rollout_cgorl()` | 相同逻辑，使用新的agent接口 |

---

## 八、使用方法

```bash
# 变体1：CURL + 弱KL（推荐）
python scripts/run_cgorl.py \
    --env_name CheetahRun \
    --kl_coeff 0.001 \
    --num_stages 4

# 变体2：CURL only
python scripts/run_cgorl.py \
    --env_name CheetahRun \
    --kl_coeff 0.0 \
    --num_stages 4

# 自定义CURL参数
python scripts/run_cgorl.py \
    --env_name HumanoidWalk \
    --curl_latent_dim 64 \
    --curl_coeff 0.5 \
    --curl_temperature 0.2 \
    --augmentation_scale 0.02
```

---

## 九、预期改进

| 指标 | 原始GoRL | C-GoRL (预期) |
|-----|---------|--------------|
| Stage间性能连续性 | 跳变 | 平滑 |
| 算力效率 | 每stage从头学 | 继承知识 |
| 表征稳定性 | 依赖KL硬约束 | CURL软约束 |
| 超参数敏感度 | 对KL系数敏感 | 更鲁棒 |

---

## 十、待验证问题

1. **CURL在state-based RL中的有效性**
   - 原始CURL针对pixel-based设计
   - 需要验证高斯噪声增强是否足够

2. **变体1 vs 变体2**
   - 弱KL是否必要？
   - CURL是否足以防止分布漂移？

3. **最优超参数**
   - `curl_latent_dim`: 50? 100?
   - `curl_temperature`: 0.1? 0.5?
   - `augmentation_scale`: 0.01? 0.1?

4. **与原始GoRL的对比**
   - 需要在多个环境上进行对比实验
   - 关注最终性能和训练稳定性

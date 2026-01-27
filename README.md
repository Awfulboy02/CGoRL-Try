# C-GoRL: Contrastive Generative Online RL (Experimental)

这是一个基于 GoRL (Generative Online RL) 的改进版本。本项目的主要目的是解决原版 GoRL 中 **Decoder 更新导致非平稳性，从而被迫重置 Encoder** 的低效问题。

我们引入了 **CURL (Contrastive Unsupervised Representations for Learning)** 来构建语义稳定的状态表征，并在此基础上实现了 **无重置 (No-Reset)** 的持续训练架构。

> ⚠️ **Status**: 目前代码完成度较低，实验结果仍在优化中。

## 核心改进机制 (Key Mechanisms)

相对于原版 GoRL，本代码库引入了以下核心改动：

### 持续进化的 Encoder
* **原版 GoRL**: 每个 Stage 必须重置 Encoder 参数，导致丢弃已学特征。
* **C-GoRL**: Encoder 参数在 Stage 之间**全量继承**。通过引入辅助任务 (Contrastive Loss) 锚定特征空间，使其对 Decoder 的变化保持鲁棒。

---

## 快速开始 (Usage)

### 1. 环境准备
请确保已安装 `mujoco` 相关依赖。
```bash
pip install -r requirements.txt

python scripts/run_cgorl.py \
    --env_name HopperStand \
    --seed 1 \
    --kl_coeff 0.01

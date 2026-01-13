#!/bin/bash

# ==============================================================================
# GoRL (Flow Matching) 论文复现脚本
# 论文: GORL: An Algorithm-Agnostic Framework for Online RL with Generative Policies
# 环境: DeepMind Control Suite (6个核心任务)
# 设置: 4 Stages, Total 180M Steps (60M+60M+30M+30M)
# 种子: 5个随机种子 (1-5)
# ==============================================================================

# 1. 定义论文中的6个环境 
ENVS=( "FingerSpin" "FingerTurnHard" "FishSwim" "HopperStand" "WalkerWalk")
# "CheetahRun"
# 2. 定义随机种子 (论文结果基于5个种子取平均) 
SEEDS=(1 2 3 4 5)
# SEEDS=(1)
# 3. 定义训练参数
# 论文使用 4 个阶段，步数分配为 60M, 60M, 30M, 30M 
NUM_STAGES=4
TIMESTEPS="60000000,60000000,30000000,30000000"

# 指定使用的 GPU ID (根据你的机器修改，例如 0 或 0,1)
GPU_ID=0

# 创建日志目录用于保存终端输出（除了代码内部的results日志外，额外备份一份标准输出）
mkdir -p logs

echo "=========================================================="
echo "开始 GoRL (FM) 全量实验复现"
echo "环境列表: ${ENVS[*]}"
echo "种子列表: ${SEEDS[*]}"
echo "总步数: 180M (分4阶段: $TIMESTEPS)"
echo "=========================================================="

# 循环环境
for env in "${ENVS[@]}"; do
    echo ""
    echo ">>> 进入环境: $env"
    
    # 循环种子
    for seed in "${SEEDS[@]}"; do
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="logs/${env}_seed${seed}_${timestamp}.log"
        
        echo "    [$(date '+%H:%M:%S')] 开始训练: Env=$env, Seed=$seed, Stages=$NUM_STAGES"
        echo "    日志记录在: $log_file (以及 results/ 目录下)"

        # 执行训练命令
        # 注意：我们不手动设置 z_regularization，让脚本使用默认的自适应逻辑
        # (Stage 0 使用较低正则化，Stage 1+ 使用 0.001，这符合代码的最佳实践)
        python scripts/run_gorl_fm.py \
            --env_name "$env" \
            --num_stages $NUM_STAGES \
            --encoder_timesteps_per_stage "$TIMESTEPS" \
            --seed $seed \
            > "$log_file" 2>&1

        # 检查上一条命令是否成功
        if [ $? -eq 0 ]; then
            echo "    [$(date '+%H:%M:%S')] 成功: $env (Seed $seed) 完成."
        else
            echo "    [$(date '+%H:%M:%S')] 失败: $env (Seed $seed) 遇到错误，查看 $log_file"
        fi
        
    done
done

echo ""
echo "=========================================================="
echo "所有实验已结束。"
echo "数据已保存至 results/ 目录，可用于画图。"
echo "=========================================================="
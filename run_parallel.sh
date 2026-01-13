#!/bin/bash

# ==============================================================================
# GoRL 多卡并行训练脚本
# 目标: 并行运行 5 个环境，每个环境绑定一张独立显卡
# ==============================================================================

# 定义配置
NUM_STAGES=4
TIMESTEPS="60000000,60000000,30000000,30000000"
SEEDS=(1)

# 创建日志目录
mkdir -p logs

# 定义一个函数，用于在指定GPU上运行特定环境的所有种子
run_env_on_gpu() {
    local gpu_id=$1
    local env_name=$2
    
    echo ">>> [GPU $gpu_id] 启动任务: $env_name (将在后台顺序运行 5 个种子)"
    
    for seed in "${SEEDS[@]}"; do
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="logs/${env_name}_seed${seed}_${timestamp}.log"
        
        echo "    [GPU $gpu_id] 开始: $env_name (Seed $seed) -> 日志: $log_file"
        
        # 核心命令：指定 CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES=$gpu_id python scripts/run_cgorl.py \
            --env_name "$env_name" \
            --num_stages $NUM_STAGES \
            --seed $seed \
            > "$log_file" 2>&1
            
        if [ $? -eq 0 ]; then
            echo "    [GPU $gpu_id] 完成: $env_name (Seed $seed)"
        else
            echo "    [GPU $gpu_id] 错误: $env_name (Seed $seed) 失败，请检查日志"
        fi
    done
    
    echo ">>> [GPU $gpu_id] $env_name 的所有种子已完成！"
}

# ==============================================================================
# 任务分配区
# 格式: run_env_on_gpu [显卡ID] [环境名] &
# 注意末尾的 '&' 符号，它让任务在后台运行，从而实现并行
# ==============================================================================

# 注意避开忙碌的 GPU
#ENVS=("CheetahRun" "FingerSpin" "FingerTurnHard" "FishSwim" "HopperStand" "WalkerWalk")
run_env_on_gpu 0 "CheetahRun" &
run_env_on_gpu 1 "FingerSpin" &
run_env_on_gpu 5 "FishSwim" &
run_env_on_gpu 6 "HopperStand" &
run_env_on_gpu 7 "WalkerWalk" &

# ==============================================================================

# 等待所有后台任务结束
echo "所有任务已在后台启动..."
echo "你可以使用 'tail -f logs/*.log' 查看实时进度"
echo "或使用 'nvidia-smi' 查看显卡占用情况"
wait
echo "所有并行任务全部执行完毕！"
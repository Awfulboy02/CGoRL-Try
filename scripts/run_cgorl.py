#!/usr/bin/env python3
"""C-GoRL Training Pipeline.

This script implements the full C-GoRL training loop:
- Stage 0: Init decoder → Phase 1 (CURL+PPO) → Collect data → Phase 2 (Decoder)
- Stage N: Phase 1 (inherit encoder!) → Collect data → Phase 2

Key difference from original GoRL:
- NO encoder reset between stages
- CURL provides representation stability
- Optional weak KL regularization (Variant 1 vs Variant 2)

Usage:
    # Variant 1: CURL + weak KL (recommended)
    python scripts/run_cgorl.py --env_name CheetahRun --kl_coeff 0.001
    
    # Variant 2: CURL only, no KL
    python scripts/run_cgorl.py --env_name CheetahRun --kl_coeff 0.0
"""

import datetime
import pickle
import sys
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import tyro
from mujoco_playground import registry

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flow_policy.cgorl import (
    CGoRLConfig,
    CGoRLEncoderState,
    CGoRLAgent,
    CGoRLRolloutState,
    eval_cgorl_policy,
    collect_data_for_decoder,
)
from flow_policy.decoder_fm import DecoderFMState, DecoderFMConfig


def main(
    # Environment
    env_name: Annotated[str, tyro.conf.arg(help="Environment name")] = "CheetahRun",
    seed: int = 1,
    
    # Training stages
    num_stages: Annotated[int, tyro.conf.arg(help="Number of training stages")] = 4,
    encoder_timesteps_per_stage: Annotated[
        str, tyro.conf.arg(help="Comma-separated timesteps per stage")
    ] = "60000000,60000000,30000000,30000000",
    
    # C-GoRL specific
    curl_coeff: float = 1.0,         # λ1: CURL loss weight
    kl_coeff: float = 0.001,         # λ2: KL loss weight (0 for Variant 2/3)
    curl_latent_dim: int = 50,       # CURL representation dimension
    curl_momentum: float = 0.95,     # EMA momentum
    curl_temperature: float = 0.1,   # InfoNCE temperature
    augmentation_scale: float = 0.01,# State augmentation noise
    
    # =========================================================================
    # 改进方案参数 (C-GoRL v2)
    # =========================================================================
    # 方案A: Policy Head Reset - 在Stage切换时重置策略头，恢复高熵探索
    reset_policy_head: bool = False,  # 是否在Stage切换时重置Policy网络
    
    # 方案B: Tanh Squashing - 硬性分布约束，消除OOD问题
    use_tanh_squashing: bool = False, # 是否启用tanh约束
    latent_scale: float = 3.0,        # tanh输出的缩放系数
    
    # PPO parameters
    learning_rate: float = 1e-3,
    clipping_epsilon: float = 0.2,
    max_grad_norm: float = 0.5,
    entropy_cost: float = 0.01,
    
    # Decoder parameters
    fm_batch_size: int = 8192,
    fm_num_epochs: int = 50,
    fm_learning_rate: float = 3e-4,
    fm_hidden_size: int = 64,
    fm_num_layers: int = 4,
    
    # Data collection
    data_collection_iterations: int = 20,
    
    # Evaluation (与原GoRL保持一致的评估频率)
    eval_frequency: int = 1_000_000,  # 100万步评估一次，与原GoRL一致
    num_eval_envs: int = 128,
    
) -> None:
    """Run C-GoRL training pipeline.
    
    改进版本支持:
    - 方案A (reset_policy_head): Stage切换时重置Policy头，解决探索僵化
    - 方案B (use_tanh_squashing): Tanh约束潜空间，解决分布失配和震荡
    """
    
    # Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建variant名称，反映所有配置选项
    variant_parts = []
    if use_tanh_squashing:
        variant_parts.append("tanh")
    elif kl_coeff > 0:
        variant_parts.append("weakKL")
    else:
        variant_parts.append("noKL")
    
    if reset_policy_head:
        variant_parts.append("resetHead")
    
    variant = "_".join(variant_parts)
    
    run_id = f"cgorl_{variant}_{env_name}_seed{seed}_{timestamp}"
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    prng = jax.random.PRNGKey(seed)
    
    # Parse timesteps per stage
    timesteps_list = [int(x.strip()) for x in encoder_timesteps_per_stage.split(",")]
    if len(timesteps_list) < num_stages:
        timesteps_list.extend([timesteps_list[-1]] * (num_stages - len(timesteps_list)))
    timesteps_list = timesteps_list[:num_stages]
    
    # Load environment
    env_config = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_config)
    obs_dim = env.observation_size
    action_dim = env.action_size
    
    print(f"\n{'='*60}", flush=True)
    print(f"C-GoRL Training - {variant}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Environment: {env_name}", flush=True)
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}", flush=True)
    print(f"Stages: {num_stages}, Timesteps: {timesteps_list}", flush=True)
    print(f"CURL coeff: {curl_coeff}, KL coeff: {kl_coeff}", flush=True)
    # 打印改进方案配置
    if use_tanh_squashing:
        print(f"[改进B] Tanh Squashing: enabled (scale={latent_scale})", flush=True)
    if reset_policy_head:
        print(f"[改进A] Policy Head Reset: enabled", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Save config
    config_file = run_dir / "config.txt"
    with open(config_file, "w") as f:
        f.write(f"C-GoRL Configuration ({variant})\n")
        f.write(f"{'='*50}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n")
        f.write(f"Timesteps: {timesteps_list}\n")
        f.write(f"CURL coeff: {curl_coeff}\n")
        f.write(f"KL coeff: {kl_coeff}\n")
        f.write(f"CURL latent dim: {curl_latent_dim}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"\n--- 改进方案配置 ---\n")
        f.write(f"Policy Head Reset: {reset_policy_head}\n")
        f.write(f"Tanh Squashing: {use_tanh_squashing}\n")
        f.write(f"Latent Scale: {latent_scale}\n")
    
    # Initialize tracking
    encoder_checkpoint = None
    decoder_checkpoint = None
    global_metrics_file = run_dir / "eval_metrics.txt"
    all_stage_metrics = {}  # 保存所有Stage的metrics用于画图
    cumulative_step = 0
    
    with open(global_metrics_file, "w") as f:
        f.write(f"C-GoRL Evaluation Metrics ({variant})\n")
        f.write(f"{'='*50}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n")
        f.write(f"Timesteps per stage: {timesteps_list}\n")
        f.write(f"Eval frequency: {eval_frequency}\n")
        f.write(f"CURL coeff: {curl_coeff}\n")
        f.write(f"KL coeff: {kl_coeff}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"{'='*50}\n\n")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    
    for stage in range(num_stages):
        print(f"\n{'='*60}", flush=True)
        print(f"STAGE {stage}/{num_stages-1}", flush=True)
        print(f"{'='*60}", flush=True)
        
        stage_dir = run_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        stage_timesteps = timesteps_list[stage]
        prng, stage_prng = jax.random.split(prng)
        
        # ======================================================================
        # Phase 0: Initialize/Load Decoder (Stage 0 only: identity decoder)
        # ======================================================================
        
        if stage == 0:
            print("\n[Phase 0] Initializing identity decoder...", flush=True)
            decoder_state = _init_identity_decoder(
                stage_prng, obs_dim, action_dim, fm_hidden_size, fm_num_layers
            )
            decoder_checkpoint = stage_dir / "decoder_init.pkl"
            _save_decoder(decoder_state, decoder_checkpoint)
            print(f"  Saved: {decoder_checkpoint}", flush=True)
        else:
            print(f"\n[Phase 0] Loading decoder from previous stage...", flush=True)
            decoder_state = _load_decoder(decoder_checkpoint)
        
        # ======================================================================
        # Phase 1: Train Encoder (CURL + PPO)
        # ======================================================================
        
        print(f"\n[Phase 1] Training encoder ({stage_timesteps:,} steps)...", flush=True)
        
        # Create config for this stage
        config = CGoRLConfig(
            z_dim=action_dim,
            curl_latent_dim=curl_latent_dim,
            curl_coeff=curl_coeff,
            kl_coeff=kl_coeff,
            curl_momentum=curl_momentum,
            curl_temperature=curl_temperature,
            augmentation_scale=augmentation_scale,
            learning_rate=learning_rate,
            clipping_epsilon=clipping_epsilon,
            max_grad_norm=max_grad_norm,
            entropy_cost=entropy_cost,
            num_timesteps=stage_timesteps,
            # 改进方案B: Tanh Squashing
            use_tanh_squashing=use_tanh_squashing,
            latent_scale=latent_scale,
        )
        
        # Initialize or load encoder
        if stage == 0 or encoder_checkpoint is None:
            print("  Initializing encoder from scratch...", flush=True)
            prng, init_prng = jax.random.split(prng)
            encoder_state = CGoRLEncoderState.init(init_prng, obs_dim, config)
        else:
            print(f"  Loading encoder from: {encoder_checkpoint}", flush=True)
            encoder_state = CGoRLEncoderState.load(str(encoder_checkpoint), obs_dim, config)
            
            # ================================================================
            # 改进方案A: Policy Head Reset
            # 在Stage切换时重置Policy网络，保留CURL Encoder
            # ================================================================
            if reset_policy_head:
                print("  [Policy Head Reset] Resetting policy network for new stage exploration...", flush=True)
                prng, reset_prng = jax.random.split(prng)
                encoder_state = encoder_state.reset_policy_head(reset_prng)
                print("    - CURL encoder: preserved", flush=True)
                print("    - Policy head: reset to high-entropy state", flush=True)
        
        # Create agent
        agent = CGoRLAgent(encoder_state=encoder_state, decoder_state=decoder_state)
        
        # Training loop
        prng, train_prng = jax.random.split(prng)
        agent, stage_metrics = _train_encoder_phase1(
            agent=agent,
            env=env,
            config=config,
            prng=train_prng,
            stage_timesteps=stage_timesteps,
            eval_frequency=eval_frequency,
            num_eval_envs=num_eval_envs,
            stage_dir=stage_dir,
            global_metrics_file=global_metrics_file,
            cumulative_step=cumulative_step,
            stage=stage,  # 传入当前Stage编号
        )
        
        cumulative_step += stage_timesteps
        
        # Save encoder checkpoint
        encoder_checkpoint = stage_dir / "encoder_checkpoint.pkl"
        agent.encoder_state.save(str(encoder_checkpoint))
        print(f"  Saved encoder: {encoder_checkpoint}", flush=True)
        
        # ======================================================================
        # Phase 2: Collect Data
        # ======================================================================
        
        print(f"\n[Phase 2] Collecting data for decoder training...", flush=True)
        
        # 计算合理的收集迭代数
        # 目标：最多收集 max_samples 个样本
        max_samples = 5_000_000  # 500万样本，足够训练decoder
        target_iterations = max_samples // config.num_envs
        actual_iterations = min(
            data_collection_iterations * config.episode_length,
            target_iterations
        )
        expected_samples = actual_iterations * config.num_envs
        print(f"  Target iterations: {actual_iterations:,}, Expected samples: {expected_samples:,}", flush=True)
        
        prng, collect_prng = jax.random.split(prng)
        obs_data, action_data, reward_data = collect_data_for_decoder(
            agent=agent,
            env=env,
            prng=collect_prng,
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            num_iterations=actual_iterations,
        )
        
        # Flatten and save
        obs_flat = obs_data.reshape(-1, obs_dim)
        action_flat = action_data.reshape(-1, action_dim)
        
        data_file = stage_dir / "collected_data.pkl"
        with open(data_file, "wb") as f:
            pickle.dump({
                "obs": obs_flat,
                "actions": action_flat,
                "env_name": env_name,
            }, f)
        print(f"  Collected {obs_flat.shape[0]:,} samples, saved: {data_file}", flush=True)
        
        # ======================================================================
        # Phase 3: Train Decoder
        # ======================================================================
        
        print(f"\n[Phase 3] Training decoder...", flush=True)
        
        prng, decoder_prng = jax.random.split(prng)
        decoder_state = _train_decoder_phase2(
            obs_data=obs_flat,
            action_data=action_flat,
            prng=decoder_prng,
            obs_dim=obs_dim,
            action_dim=action_dim,
            batch_size=fm_batch_size,
            num_epochs=fm_num_epochs,
            learning_rate=fm_learning_rate,
            hidden_size=fm_hidden_size,
            num_layers=fm_num_layers,
        )
        
        # Save decoder
        decoder_checkpoint = stage_dir / "decoder_trained.pkl"
        _save_decoder(decoder_state, decoder_checkpoint)
        print(f"  Saved decoder: {decoder_checkpoint}", flush=True)
        
        # 保存Stage metrics用于画图
        all_stage_metrics[stage] = stage_metrics["metrics"]
        
        # 保存Stage级别的metrics到pickle (用于Figure 6类型的图)
        stage_metrics_file = stage_dir / "stage_metrics.pkl"
        with open(stage_metrics_file, "wb") as f:
            pickle.dump({
                "stage": stage,
                "metrics": stage_metrics["metrics"],
                "best_reward": stage_metrics["best_reward"],
                "cumulative_step_start": cumulative_step - stage_timesteps,
                "cumulative_step_end": cumulative_step,
            }, f)
        
        # Stage summary
        summary_file = stage_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Stage {stage} Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Encoder: {encoder_checkpoint}\n")
            f.write(f"Decoder: {decoder_checkpoint}\n")
            f.write(f"Data samples: {obs_flat.shape[0]}\n")
            f.write(f"Best reward: {stage_metrics['best_reward']:.2f}\n")
            f.write(f"Eval points: {len(stage_metrics['metrics'])}\n")
        
        print(f"\nStage {stage} complete!", flush=True)
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    
    print(f"\n{'='*60}", flush=True)
    print(f"C-GoRL Training Complete!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Output directory: {run_dir}", flush=True)
    print(f"Final encoder: {encoder_checkpoint}", flush=True)
    print(f"Final decoder: {decoder_checkpoint}", flush=True)
    
    # 保存完整的训练metrics到pickle文件 (用于画图)
    all_metrics_file = run_dir / "all_metrics.pkl"
    with open(all_metrics_file, "wb") as f:
        pickle.dump({
            "env_name": env_name,
            "variant": variant,
            "num_stages": num_stages,
            "timesteps_per_stage": timesteps_list,
            "total_timesteps": cumulative_step,
            "curl_coeff": curl_coeff,
            "kl_coeff": kl_coeff,
            "seed": seed,
            "eval_frequency": eval_frequency,
            "stage_metrics": all_stage_metrics,  # 按Stage组织的metrics
        }, f)
    print(f"Saved all metrics: {all_metrics_file}", flush=True)
    
    final_summary = run_dir / "final_summary.txt"
    with open(final_summary, "w") as f:
        f.write(f"C-GoRL Final Summary ({variant})\n")
        f.write(f"{'='*50}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages completed: {num_stages}\n")
        f.write(f"Total timesteps: {cumulative_step:,}\n")
        f.write(f"Eval frequency: {eval_frequency:,}\n")
        f.write(f"Total eval points: {sum(len(m) for m in all_stage_metrics.values())}\n")
        f.write(f"Final encoder: {encoder_checkpoint}\n")
        f.write(f"Final decoder: {decoder_checkpoint}\n")
        f.write(f"\nPer-stage best rewards:\n")
        for s, metrics in all_stage_metrics.items():
            if metrics:
                best = max(m["reward_mean"] for m in metrics)
                f.write(f"  Stage {s}: {best:.2f}\n")


# =============================================================================
# Helper Functions
# =============================================================================

def _init_identity_decoder(
    prng: jax.Array,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    num_layers: int,
) -> DecoderFMState:
    """Initialize identity decoder (maps z → z approximately)."""
    from flow_policy.math_utils import RunningStats
    
    config = DecoderFMConfig(
        hidden_dims=tuple([hidden_size] * num_layers),
        learning_rate=3e-4,
    )
    
    state = DecoderFMState.init(prng, obs_dim, action_dim, config)
    
    # Set observation stats to identity
    state = jdc.replace(
        state,
        obs_stats=RunningStats.init((obs_dim,)),
    )
    
    return state


def _save_decoder(state: DecoderFMState, path: Path) -> None:
    """Save decoder state."""
    with open(path, "wb") as f:
        pickle.dump({
            "params": state.params,
            "obs_stats": state.obs_stats,
            "config": state.config,
        }, f)


def _load_decoder(path: Path) -> DecoderFMState:
    """Load decoder state."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    # Reconstruct state
    config = checkpoint["config"]
    prng = jax.random.PRNGKey(0)
    
    # Get dimensions from params
    input_dim = checkpoint["params"][0][0].shape[0]
    output_dim = checkpoint["params"][-1][0].shape[1]
    obs_dim = input_dim - output_dim - config.timestep_embed_dim
    
    state = DecoderFMState.init(prng, obs_dim, output_dim, config)
    state = jdc.replace(
        state,
        params=checkpoint["params"],
        obs_stats=checkpoint["obs_stats"],
    )
    
    return state


def _train_encoder_phase1(
    agent: CGoRLAgent,
    env,
    config: CGoRLConfig,
    prng: jax.Array,
    stage_timesteps: int,
    eval_frequency: int,
    num_eval_envs: int,
    stage_dir: Path,
    global_metrics_file: Path,
    cumulative_step: int,
    stage: int,  # 新增：当前Stage编号
) -> tuple[CGoRLAgent, dict]:
    """Phase 1: Train encoder with CURL + PPO."""
    
    # 在Stage开始时写入标记
    with open(global_metrics_file, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"STAGE {stage}\n")
        f.write(f"{'='*50}\n\n")
    
    # Initialize rollout state
    prng, rollout_prng = jax.random.split(prng)
    rollout_state = CGoRLRolloutState.init(env, rollout_prng, config.num_envs)
    
    # Training metrics
    best_reward = float("-inf")
    all_metrics = []
    
    # Calculate number of training iterations
    steps_per_iter = config.iterations_per_env * config.num_envs
    num_iterations = stage_timesteps // steps_per_iter
    
    # =========================================================================
    # 计算评估点（与原GoRL一致的迭代间隔法）
    # =========================================================================
    # 每隔 eval_step_interval 次迭代评估一次
    # 例如: eval_frequency=1M, steps_per_iter=163,840 → 每6次迭代评估一次
    eval_step_interval = max(1, eval_frequency // steps_per_iter)
    eval_iters = set(range(0, num_iterations, eval_step_interval))
    eval_iters.add(num_iterations - 1)  # 确保最后一次迭代也评估
    
    # 计算实际评估间隔步数
    actual_eval_steps = eval_step_interval * steps_per_iter
    
    print(f"  Steps per iteration: {steps_per_iter:,}", flush=True)
    print(f"  Total iterations: {num_iterations:,}", flush=True)
    print(f"  Eval every {eval_step_interval} iters (~{actual_eval_steps:,} steps), total {len(eval_iters)} eval points", flush=True)
    
    # Progress printing frequency
    print_frequency = max(1, num_iterations // 100)  # Print ~100 times per stage
    
    for iteration in range(num_iterations):
        current_step = iteration * steps_per_iter
        global_step = cumulative_step + current_step
        
        # Progress indicator
        if iteration % print_frequency == 0 and iteration > 0:
            progress = iteration / num_iterations * 100
            print(f"    Progress: {progress:.1f}% ({iteration}/{num_iterations})", flush=True)
        
        # Rollout
        prng, rollout_prng = jax.random.split(prng)
        rollout_state = jdc.replace(rollout_state, prng=rollout_prng)
        
        rollout_state, transitions = rollout_state.rollout(
            agent=agent,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
            deterministic=False,
        )
        
        # Training step
        agent, train_metrics = agent.training_step(transitions)
        
        # Evaluation (使用预计算的eval_iters集合，与原GoRL一致)
        if iteration in eval_iters:
            prng, eval_prng = jax.random.split(prng)
            eval_metrics = eval_cgorl_policy(
                agent=agent,
                env=env,
                prng=eval_prng,
                num_envs=num_eval_envs,
                max_episode_length=config.episode_length,
            )
            
            # Extract metrics
            reward_mean = float(eval_metrics["reward_mean"])
            reward_std = float(eval_metrics["reward_std"])
            reward_min = float(eval_metrics["reward_min"])
            reward_max = float(eval_metrics["reward_max"])
            steps_mean = float(eval_metrics["steps_mean"])
            steps_std = float(eval_metrics["steps_std"])
            steps_min = float(eval_metrics["steps_min"])
            steps_max = float(eval_metrics["steps_max"])
            
            # Console output
            print(f"  Step {global_step:>10,}: reward={reward_mean:.1f}±{reward_std:.1f} "
                  f"(curl={float(train_metrics['curl_loss'][-1,-1]):.4f}, "
                  f"kl={float(train_metrics['kl_loss'][-1,-1]):.4f})", flush=True)
            
            # Log to file (格式与原GoRL兼容)
            with open(global_metrics_file, "a") as f:
                f.write(f"Step: {global_step}\n")
                f.write(f"-" * 50 + "\n")
                # Reward metrics (核心指标，用于画图)
                f.write(f"reward_mean: {reward_mean:.4f}\n")
                f.write(f"reward_std: {reward_std:.4f}\n")
                f.write(f"reward_min: {reward_min:.4f}\n")
                f.write(f"reward_max: {reward_max:.4f}\n")
                # Steps metrics
                f.write(f"steps_mean: {steps_mean:.4f}\n")
                f.write(f"steps_std: {steps_std:.4f}\n")
                f.write(f"steps_min: {steps_min:.4f}\n")
                f.write(f"steps_max: {steps_max:.4f}\n")
                # C-GoRL specific metrics (CURL相关)
                f.write(f"curl_loss: {float(train_metrics['curl_loss'][-1,-1]):.6f}\n")
                f.write(f"kl_loss: {float(train_metrics['kl_loss'][-1,-1]):.6f}\n")
                # PPO metrics
                f.write(f"policy_loss: {float(train_metrics['policy_loss'][-1,-1]):.6f}\n")
                f.write(f"value_loss: {float(train_metrics['value_loss'][-1,-1]):.6f}\n")
                f.write(f"entropy: {float(train_metrics['entropy'][-1,-1]):.4f}\n")
                # Latent space statistics
                f.write(f"eps_mean: {float(train_metrics['eps_mean'][-1,-1]):.4f}\n")
                f.write(f"eps_std: {float(train_metrics['eps_std'][-1,-1]):.4f}\n")
                f.write(f"z_s_mean: {float(train_metrics['z_s_mean'][-1,-1]):.4f}\n")
                f.write(f"z_s_std: {float(train_metrics['z_s_std'][-1,-1]):.4f}\n")
                f.write("\n")
            
            # Save best
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_checkpoint = stage_dir / "best_encoder.pkl"
                agent.encoder_state.save(str(best_checkpoint))
            
            all_metrics.append({
                "step": global_step,
                "local_step": current_step,  # 用于Figure 6类型的图
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "steps_mean": steps_mean,
                **{k: float(v[-1,-1]) for k, v in train_metrics.items()},
            })
    
    return agent, {"best_reward": best_reward, "metrics": all_metrics}


def _train_decoder_phase2(
    obs_data: jax.Array,
    action_data: jax.Array,
    prng: jax.Array,
    obs_dim: int,
    action_dim: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
) -> DecoderFMState:
    """Phase 2: Train decoder with supervised learning."""
    from flow_policy.math_utils import RunningStats
    
    config = DecoderFMConfig(
        hidden_dims=tuple([hidden_size] * num_layers),
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    
    prng, init_prng = jax.random.split(prng)
    decoder_state = DecoderFMState.init(init_prng, obs_dim, action_dim, config)
    
    # Update observation statistics incrementally to save memory
    obs_stats = RunningStats.init((obs_dim,))
    num_samples = obs_data.shape[0]
    stats_batch_size = min(100000, num_samples)  # 分批更新统计量
    
    for i in range(0, num_samples, stats_batch_size):
        end_idx = min(i + stats_batch_size, num_samples)
        obs_stats = obs_stats.update(obs_data[i:end_idx])
    
    decoder_state = jdc.replace(decoder_state, obs_stats=obs_stats)
    
    # Training
    num_batches = num_samples // batch_size
    
    print(f"  Samples: {num_samples:,}, Batches: {num_batches}, Epochs: {num_epochs}", flush=True)
    
    for epoch in range(num_epochs):
        # Shuffle
        prng, shuffle_prng = jax.random.split(prng)
        perm = jax.random.permutation(shuffle_prng, num_samples)
        obs_shuffled = obs_data[perm]
        action_shuffled = action_data[perm]
        
        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            
            batch_obs = obs_shuffled[start:end]
            batch_actions = action_shuffled[start:end]
            
            decoder_state, metrics = decoder_state.train_step(batch_obs, batch_actions)
            epoch_loss += float(metrics["loss"])
        
        avg_loss = epoch_loss / num_batches
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch:>3}: loss={avg_loss:.6f}", flush=True)
    
    return decoder_state


if __name__ == "__main__":
    tyro.cli(main)

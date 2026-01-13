"""C-GoRL: Contrastive GoRL - CURL + GoRL without encoder reset.

This module implements C-GoRL which replaces the original GoRL's 
"reset encoder + KL align to N(0,1)" strategy with CURL-based 
contrastive representation learning.

Key differences from original GoRL:
1. CURL encoder f_ω maps obs → z_s (stable representation)
2. Latent policy π_θ(ε|z_s) conditions on CURL representation
3. NO encoder reset between stages - parameters are inherited
4. InfoNCE loss provides representation stability instead of KL reset

Architecture:
    obs → f_ω(obs) → z_s → π_θ(ε|z_s) → ε → g_φ(obs, ε) → action
          ~~~~~~~        ~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~
          CURL encoder   Latent policy      Decoder (frozen)
"""

from __future__ import annotations

from typing import NamedTuple
import pickle
from pathlib import Path

import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp

# Import from original GoRL (no modifications needed)
from flow_policy.networks import MlpWeights, mlp_init, gaussian_policy_fwd, value_mlp_fwd
from flow_policy.math_utils import RunningStats, NormalDistribution
from flow_policy.decoder_fm import DecoderFMState
from flow_policy import rollouts


# =============================================================================
# Configuration
# =============================================================================

@jdc.pytree_dataclass
class CGoRLConfig:
    """Configuration for C-GoRL.
    
    Variant 1 (CURL + weak KL): kl_coeff > 0 (e.g., 0.001)
    Variant 2 (CURL + no KL):   kl_coeff = 0
    """
    # Environment
    action_repeat: jdc.Static[int] = 1
    episode_length: int = 1000
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 128
    
    # Dimensions
    z_dim: jdc.Static[int] = 6  # Latent action dimension (= action_dim)
    curl_latent_dim: jdc.Static[int] = 50  # CURL representation dimension
    
    # CURL parameters
    curl_hidden_dims: jdc.Static[tuple[int, ...]] = (256, 256)
    curl_momentum: float = 0.95  # EMA momentum for key encoder
    curl_coeff: float = 1.0  # λ1: CURL loss coefficient
    curl_temperature: float = 0.1  # InfoNCE temperature
    augmentation_scale: float = 0.01  # Gaussian noise for state augmentation
    
    # KL regularization (Variant 1 vs Variant 2)
    kl_coeff: float = 0.001  # λ2: KL loss coefficient (0 for Variant 2)
    
    # PPO parameters (from original encoder_ppo.py)
    batch_size: jdc.Static[int] = 256
    num_minibatches: jdc.Static[int] = 32
    num_updates_per_batch: jdc.Static[int] = 4
    unroll_length: jdc.Static[int] = 20
    num_timesteps: jdc.Static[int] = 100_000_000
    
    learning_rate: float = 1e-3
    discounting: float = 0.99
    gae_lambda: float = 0.95
    entropy_cost: float = 0.01
    value_loss_coeff: float = 0.25
    clipping_epsilon: float = 0.2
    max_grad_norm: jdc.Static[float] = 0.5
    reward_scaling: float = 1.0
    
    normalize_observations: jdc.Static[bool] = True
    normalize_advantage: jdc.Static[bool] = True
    
    @property
    def iterations_per_env(self) -> int:
        return (self.num_minibatches * self.batch_size * self.unroll_length) // self.num_envs


# =============================================================================
# CURL Encoder
# =============================================================================

@jdc.pytree_dataclass
class CURLEncoderState:
    """CURL representation encoder with momentum-averaged key encoder.
    
    Implements InfoNCE contrastive learning for stable representations.
    """
    query_params: MlpWeights  # f_ω (online encoder)
    key_params: MlpWeights    # f_ω^EMA (momentum encoder)
    W: Array                  # Bilinear matrix for similarity
    
    @staticmethod
    def init(prng: Array, obs_dim: int, latent_dim: int, 
             hidden_dims: tuple[int, ...] = (256, 256)) -> CURLEncoderState:
        """Initialize CURL encoder."""
        prng1, prng2, prng3 = jax.random.split(prng, 3)
        
        # Build encoder network: obs → latent
        layer_dims = (obs_dim,) + hidden_dims + (latent_dim,)
        query_params = mlp_init(prng1, layer_dims)
        key_params = mlp_init(prng2, layer_dims)  # Same init, will diverge via EMA
        
        # Bilinear matrix for similarity: sim(q, k) = q^T W k
        W = jax.random.normal(prng3, (latent_dim, latent_dim)) * 0.01
        
        return CURLEncoderState(
            query_params=query_params,
            key_params=key_params,
            W=W,
        )
    
    def encode_query(self, obs: Array) -> Array:
        """Encode observation using query encoder."""
        return _mlp_forward_tanh(self.query_params, obs)
    
    def encode_key(self, obs: Array) -> Array:
        """Encode observation using key encoder (no gradient)."""
        return jax.lax.stop_gradient(_mlp_forward_tanh(self.key_params, obs))
    
    def update_ema(self, momentum: float) -> CURLEncoderState:
        """Update key encoder with EMA of query encoder."""
        new_key_params = jax.tree.map(
            lambda q, k: momentum * k + (1 - momentum) * q,
            self.query_params,
            self.key_params,
        )
        return jdc.replace(self, key_params=new_key_params)


def _mlp_forward_tanh(weights: MlpWeights, x: Array) -> Array:
    """MLP forward pass with SiLU hidden activations and tanh output."""
    for i in range(len(weights) - 1):
        linear, bias = weights[i]
        x = jnp.einsum("...i,ij->...j", x, linear) + bias
        x = jax.nn.silu(x)
    
    # Final layer with tanh normalization (like CURL)
    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias
    x = jnp.tanh(x)
    return x


def augment_state(obs: Array, prng: Array, scale: float) -> Array:
    """State-based data augmentation using Gaussian noise.
    
    For state-based RL (not pixel-based), we use simple noise injection.
    """
    noise = jax.random.normal(prng, obs.shape) * scale
    return obs + noise


def compute_infonce_loss(
    z_query: Array, 
    z_key: Array, 
    W: Array, 
    temperature: float = 0.1
) -> Array:
    """Compute InfoNCE contrastive loss.
    
    Args:
        z_query: Query embeddings (B, D)
        z_key: Key embeddings (B, D)  
        W: Bilinear matrix (D, D)
        temperature: Softmax temperature
        
    Returns:
        InfoNCE loss scalar
    """
    # Bilinear similarity: logits[i,j] = z_q[i]^T @ W @ z_k[j]
    # Shape: (B, B)
    logits = jnp.einsum('id,de,je->ij', z_query, W, z_key) / temperature
    
    # Positive pairs are on diagonal (i == j)
    labels = jnp.arange(z_query.shape[0])
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)


# =============================================================================
# Latent Policy (based on original encoder_ppo.py)
# =============================================================================

@jdc.pytree_dataclass
class LatentPolicyActionInfo:
    """Action info for latent policy."""
    log_prob: Array
    mean: Array
    std: Array


LatentPolicyTransition = rollouts.TransitionStruct[LatentPolicyActionInfo]


@jdc.pytree_dataclass  
class LatentPolicyParams:
    """Parameters for latent policy (actor-critic)."""
    policy: MlpWeights  # z_s → (μ, σ) for ε
    value: MlpWeights   # z_s → V(s)


# =============================================================================
# C-GoRL Encoder State (CURL + Latent Policy)
# =============================================================================

@jdc.pytree_dataclass
class CGoRLEncoderState:
    """Combined C-GoRL encoder state.
    
    Contains:
    - CURL encoder for representation learning
    - Latent policy for action sampling
    - Optimizer state
    """
    config: CGoRLConfig
    curl_state: CURLEncoderState
    policy_params: LatentPolicyParams
    obs_stats: RunningStats
    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState
    prng: Array
    steps: Array
    
    @staticmethod
    def init(prng: Array, obs_dim: int, config: CGoRLConfig) -> CGoRLEncoderState:
        """Initialize C-GoRL encoder state."""
        prng1, prng2, prng3, prng4 = jax.random.split(prng, 4)
        
        # Initialize CURL encoder
        curl_state = CURLEncoderState.init(
            prng1, obs_dim, config.curl_latent_dim, config.curl_hidden_dims
        )
        
        # Initialize latent policy (input is CURL latent, not raw obs)
        # Actor: z_s → (μ, σ) for ε
        actor_net = mlp_init(
            prng2, (config.curl_latent_dim, 32, 32, 32, 32, config.z_dim * 2)
        )
        # Critic: z_s → V(s)
        critic_net = mlp_init(
            prng3, (config.curl_latent_dim, 256, 256, 256, 256, 256, 1)
        )
        
        policy_params = LatentPolicyParams(actor_net, critic_net)
        
        # Optimizer (Adam with manual LR scaling like original)
        opt = optax.scale_by_adam()
        
        # Combine all parameters for optimization
        all_params = _pack_params(curl_state, policy_params)
        
        return CGoRLEncoderState(
            config=config,
            curl_state=curl_state,
            policy_params=policy_params,
            obs_stats=RunningStats.init((obs_dim,)),
            opt=opt,
            opt_state=opt.init(all_params),
            prng=prng4,
            steps=jnp.zeros((), dtype=jnp.int32),
        )
    
    @staticmethod
    def load(checkpoint_path: str, obs_dim: int, config: CGoRLConfig) -> CGoRLEncoderState:
        """Load encoder state from checkpoint (for stage continuation)."""
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        # Create fresh state then replace with loaded params
        prng = jax.random.PRNGKey(0)
        state = CGoRLEncoderState.init(prng, obs_dim, config)
        
        # Replace with loaded parameters
        state = jdc.replace(
            state,
            curl_state=checkpoint["curl_state"],
            policy_params=checkpoint["policy_params"],
            obs_stats=checkpoint["obs_stats"],
            prng=checkpoint.get("prng", prng),
            steps=checkpoint.get("steps", jnp.zeros((), dtype=jnp.int32)),
        )
        
        # Re-initialize optimizer state for new parameters
        all_params = _pack_params(state.curl_state, state.policy_params)
        state = jdc.replace(state, opt_state=state.opt.init(all_params))
        
        return state
    
    def save(self, path: str) -> None:
        """Save encoder state to checkpoint."""
        checkpoint = {
            "curl_state": self.curl_state,
            "policy_params": self.policy_params,
            "obs_stats": self.obs_stats,
            "prng": self.prng,
            "steps": self.steps,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
    
    def sample_epsilon(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, LatentPolicyActionInfo]:
        """Sample ε from latent policy: obs → z_s → π_θ(ε|z_s) → ε.
        
        This is the core inference path for C-GoRL.
        """
        # Step 1: Normalize observation
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        else:
            obs_norm = obs
        
        # Step 2: Get CURL representation
        z_s = self.curl_state.encode_query(obs_norm)
        
        # Step 3: Sample ε from latent policy conditioned on z_s
        eps_dist = gaussian_policy_fwd(self.policy_params.policy, z_s)
        
        if deterministic:
            eps = eps_dist.loc
            log_prob = jnp.zeros_like(eps[..., 0])
        else:
            eps = eps_dist.sample(prng)
            log_prob = jnp.sum(eps_dist.log_prob(eps), axis=-1)
        
        return eps, LatentPolicyActionInfo(
            log_prob=log_prob,
            mean=eps_dist.loc,
            std=eps_dist.scale,
        )
    
    @jdc.jit
    def training_step(
        self, transitions: LatentPolicyTransition
    ) -> tuple[CGoRLEncoderState, dict[str, Array]]:
        """One training step: update CURL encoder and latent policy jointly."""
        config = self.config
        
        # Update observation statistics
        if config.normalize_observations:
            with jdc.copy_and_mutate(self) as state:
                state.obs_stats = self.obs_stats.update(transitions.obs)
        else:
            state = self
        del self
        
        def step_batch(state: CGoRLEncoderState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                lambda s, mb: s._step_minibatch(
                    mb, entropy_prng=jax.random.fold_in(step_prng, s.steps)
                ),
                init=state,
                xs=transitions.prepare_minibatches(
                    step_prng, config.num_minibatches, config.batch_size
                ),
            )
            return state, metrics
        
        state, metrics = jax.lax.scan(
            step_batch,
            init=state,
            length=config.num_updates_per_batch,
        )
        return state, metrics
    
    def _step_minibatch(
        self, transitions: LatentPolicyTransition, entropy_prng: Array
    ) -> tuple[CGoRLEncoderState, dict[str, Array]]:
        """One minibatch update."""
        
        # Compute loss and gradients
        all_params = _pack_params(self.curl_state, self.policy_params)
        
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: self._compute_combined_loss(
                params, transitions, entropy_prng
            ),
            has_aux=True,
        )(all_params)
        
        # Gradient clipping
        grad_norm = optax.global_norm(grads)
        metrics["grad_norm_before_clip"] = grad_norm
        
        if self.config.max_grad_norm > 0:
            scale = jnp.minimum(1.0, self.config.max_grad_norm / (grad_norm + 1e-8))
            grads = jax.tree.map(lambda g: g * scale, grads)
            metrics["grad_norm"] = optax.global_norm(grads)
        else:
            metrics["grad_norm"] = grad_norm
        
        # Apply updates
        param_update, new_opt_state = self.opt.update(grads, self.opt_state)
        param_update = jax.tree.map(
            lambda x: -self.config.learning_rate * x, param_update
        )
        new_params = jax.tree.map(jnp.add, all_params, param_update)
        
        # Unpack updated parameters
        new_curl_state, new_policy_params = _unpack_params(
            new_params, self.curl_state, self.policy_params
        )
        
        # Update EMA for key encoder
        new_curl_state = new_curl_state.update_ema(self.config.curl_momentum)
        
        with jdc.copy_and_mutate(self) as state:
            state.curl_state = new_curl_state
            state.policy_params = new_policy_params
            state.opt_state = new_opt_state
            state.steps = state.steps + 1
        
        return state, metrics
    
    def _compute_combined_loss(
        self,
        all_params: dict,
        transitions: LatentPolicyTransition,
        entropy_prng: Array,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute combined loss: L_PPO + λ1*L_CURL + λ2*L_KL.
        
        This is the core of C-GoRL Phase 1.
        """
        # Unpack parameters
        curl_state, policy_params = _unpack_params(
            all_params, self.curl_state, self.policy_params
        )
        
        config = self.config
        metrics = {}
        
        # Normalize observations
        if config.normalize_observations:
            obs_norm = (transitions.obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
            next_obs_norm = (transitions.next_obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        else:
            obs_norm = transitions.obs
            next_obs_norm = transitions.next_obs
        
        # =====================================================================
        # 1. CURL Loss
        # =====================================================================
        prng_aug1, prng_aug2, prng_rest = jax.random.split(entropy_prng, 3)
        
        # Reshape for augmentation (flatten batch dims)
        obs_flat = obs_norm.reshape(-1, obs_norm.shape[-1])
        batch_size_flat = obs_flat.shape[0]
        
        # Create augmented views
        obs_q = augment_state(obs_flat, prng_aug1, config.augmentation_scale)
        obs_k = augment_state(obs_flat, prng_aug2, config.augmentation_scale)
        
        # Encode
        z_query = curl_state.encode_query(obs_q)
        z_key = curl_state.encode_key(obs_k)  # No gradient through key encoder
        
        curl_loss = compute_infonce_loss(
            z_query, z_key, curl_state.W, config.curl_temperature
        )
        metrics["curl_loss"] = curl_loss
        
        # =====================================================================
        # 2. PPO Loss (in CURL latent space)
        # =====================================================================
        # Get CURL representations for policy
        z_s = curl_state.encode_query(obs_norm)
        z_s_next = curl_state.encode_query(next_obs_norm[-1:])
        
        # Value estimation
        value_pred = value_mlp_fwd(policy_params.value, z_s)
        bootstrap_value = value_mlp_fwd(policy_params.value, z_s_next)
        
        # GAE
        gae_vs, gae_advantages = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * config.discounting,
                rewards=transitions.reward * config.reward_scaling,
                values=value_pred,
                bootstrap_value=bootstrap_value,
                gae_lambda=config.gae_lambda,
            )
        )
        
        if config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (
                gae_advantages.std() + 1e-8
            )
        
        # Policy distribution in latent space
        eps_dist = gaussian_policy_fwd(policy_params.policy, z_s)
        new_log_probs = jnp.sum(eps_dist.log_prob(transitions.action), axis=-1)
        old_log_probs = transitions.action_info.log_prob
        
        # PPO clipped objective
        rho = jnp.exp(new_log_probs - old_log_probs)
        surr1 = rho * gae_advantages
        surr2 = jnp.clip(rho, 1 - config.clipping_epsilon, 1 + config.clipping_epsilon) * gae_advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Value loss
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        value_loss = jnp.mean(v_error ** 2) * config.value_loss_coeff
        
        # Entropy loss
        entropy = jnp.mean(jnp.sum(eps_dist.entropy(), axis=-1))
        entropy_loss = -config.entropy_cost * entropy
        
        metrics["policy_loss"] = policy_loss
        metrics["value_loss"] = value_loss
        metrics["entropy"] = entropy
        metrics["entropy_loss"] = entropy_loss
        
        # =====================================================================
        # 3. KL Regularization (optional, for Variant 1)
        # =====================================================================
        # KL(π_θ(ε|z_s) || N(0,I)) ≈ 0.5 * (μ² + σ² - log(σ²) - 1)
        # Simplified: just penalize μ² and σ² (entropy covers log(σ))
        kl_loss = config.kl_coeff * (
            jnp.mean(jnp.square(eps_dist.loc)) +
            jnp.mean(jnp.square(eps_dist.scale))
        )
        metrics["kl_loss"] = kl_loss
        
        # Log latent statistics
        metrics["eps_mean"] = jnp.mean(eps_dist.loc)
        metrics["eps_std"] = jnp.mean(eps_dist.scale)
        metrics["z_s_mean"] = jnp.mean(z_s)
        metrics["z_s_std"] = jnp.std(z_s)
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = (
            policy_loss + 
            value_loss + 
            entropy_loss + 
            config.curl_coeff * curl_loss +
            kl_loss  # 0 for Variant 2
        )
        metrics["total_loss"] = total_loss
        
        return total_loss, metrics


def _pack_params(curl_state: CURLEncoderState, policy_params: LatentPolicyParams) -> dict:
    """Pack all trainable parameters into a single dict for optimization."""
    return {
        "curl_query": curl_state.query_params,
        "curl_W": curl_state.W,
        "policy": policy_params.policy,
        "value": policy_params.value,
    }


def _unpack_params(
    params: dict, 
    curl_template: CURLEncoderState,
    policy_template: LatentPolicyParams
) -> tuple[CURLEncoderState, LatentPolicyParams]:
    """Unpack parameters back into structured states."""
    new_curl = jdc.replace(
        curl_template,
        query_params=params["curl_query"],
        W=params["curl_W"],
    )
    new_policy = LatentPolicyParams(
        policy=params["policy"],
        value=params["value"],
    )
    return new_curl, new_policy


# =============================================================================
# C-GoRL Agent (combines encoder + decoder)
# =============================================================================

@jdc.pytree_dataclass
class CGoRLAgent:
    """Complete C-GoRL agent with CURL encoder and FM decoder.
    
    Inference path:
        obs → CURL(obs) → z_s → π_θ(ε|z_s) → ε → g_φ(obs, ε) → action
    """
    encoder_state: CGoRLEncoderState
    decoder_state: DecoderFMState
    
    def sample_epsilon(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, LatentPolicyActionInfo]:
        """Sample ε from encoder."""
        return self.encoder_state.sample_epsilon(obs, prng, deterministic)
    
    def map_epsilon_to_action(self, obs: Array, eps: Array, prng: Array) -> Array:
        """Map ε to action using decoder."""
        return self.decoder_state.sample_action_from_z(obs, eps, prng, deterministic=True)
    
    def sample_action(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, Array, LatentPolicyActionInfo]:
        """Full inference: obs → action.
        
        Returns:
            action: Final action
            eps: Sampled ε (for storing in transitions)
            info: Action info with log_prob
        """
        prng1, prng2 = jax.random.split(prng)
        eps, info = self.sample_epsilon(obs, prng1, deterministic)
        action = self.map_epsilon_to_action(obs, eps, prng2)
        return action, eps, info
    
    def training_step(
        self, transitions: LatentPolicyTransition
    ) -> tuple[CGoRLAgent, dict[str, Array]]:
        """Update encoder only (decoder is frozen in Phase 1)."""
        new_encoder_state, metrics = self.encoder_state.training_step(transitions)
        return jdc.replace(self, encoder_state=new_encoder_state), metrics


# =============================================================================
# Rollout Functions
# =============================================================================

@jdc.pytree_dataclass
class CGoRLRolloutState:
    """Rollout state for C-GoRL."""
    env: jdc.Static  # mjp.MjxEnv (静态，不参与JIT trace)
    env_state: object  # mjp.State
    first_obs: Array
    first_data: object
    steps: Array
    num_envs: jdc.Static[int]
    prng: Array
    
    @staticmethod
    @jdc.jit
    def init(
        env: jdc.Static,
        prng: Array,
        num_envs: jdc.Static[int],
    ) -> "CGoRLRolloutState":
        """Initialize rollout state."""
        prng, reset_prng = jax.random.split(prng, num=2)
        state = jax.vmap(env.reset)(jax.random.split(reset_prng, num=num_envs))
        return CGoRLRolloutState(
            env=env,
            env_state=state,
            first_obs=state.obs,
            first_data=state.data,
            steps=jnp.zeros_like(state.done),
            num_envs=num_envs,
            prng=prng,
        )
    
    @jdc.jit
    def rollout(
        self,
        agent: CGoRLAgent,
        episode_length: jdc.Static[int],
        iterations_per_env: jdc.Static[int],
        deterministic: jdc.Static[bool] = False,
    ) -> tuple["CGoRLRolloutState", LatentPolicyTransition]:
        """Perform rollout with C-GoRL agent."""
        
        def env_step(carry: "CGoRLRolloutState", _):
            state = carry
            
            prng_sample, prng_action, prng_next = jax.random.split(state.prng, 3)
            
            # Sample action through full pipeline
            action, eps, eps_info = agent.sample_action(
                state.env_state.obs, prng_sample, deterministic
            )
            
            # Environment step
            env_action = jnp.tanh(action)
            next_env_state = jax.vmap(state.env.step)(state.env_state, env_action)
            
            # Episode management
            next_steps = state.steps + 1
            truncation = next_steps >= episode_length
            done_env = next_env_state.done.astype(bool)
            done_or_tr = jnp.logical_or(done_env, truncation)
            discount = 1.0 - done_env.astype(jnp.float32)
            
            # Store transition (eps, not action!)
            transition = rollouts.TransitionStruct(
                obs=state.env_state.obs,
                next_obs=next_env_state.obs,
                action=eps,  # Store ε for PPO update
                action_info=eps_info,
                reward=next_env_state.reward,
                truncation=truncation.astype(jnp.float32),
                discount=discount,
            )
            
            # Auto-reset
            where_done = lambda x, y: jnp.where(
                done_or_tr.reshape(done_or_tr.shape + (1,) * (x.ndim - done_or_tr.ndim)),
                x, y
            )
            next_env_state = next_env_state.replace(
                obs=jax.tree.map(where_done, state.first_obs, next_env_state.obs),
                data=jax.tree.map(where_done, state.first_data, next_env_state.data),
                done=jnp.zeros_like(next_env_state.done),
            )
            next_steps = jnp.where(done_or_tr, 0, next_steps)
            
            new_state = jdc.replace(
                state,
                env_state=next_env_state,
                steps=next_steps,
                prng=prng_next,
            )
            
            return new_state, transition
        
        final_state, transitions = jax.lax.scan(
            env_step,
            init=self,
            xs=None,
            length=iterations_per_env,
        )
        
        return final_state, transitions


def eval_cgorl_policy(
    agent: CGoRLAgent,
    env,
    prng: Array,
    num_envs: int,
    max_episode_length: int,
) -> dict[str, Array]:
    """Evaluate C-GoRL policy."""
    rollout_state = CGoRLRolloutState.init(env, prng, num_envs)
    
    _, transitions = rollout_state.rollout(
        agent=agent,
        episode_length=max_episode_length,
        iterations_per_env=max_episode_length,
        deterministic=True,
    )
    
    # Compute metrics
    rewards = jnp.sum(transitions.reward, axis=0)
    valid_mask = transitions.discount > 0.0
    steps = jnp.sum(valid_mask, axis=0)
    
    return {
        "reward_mean": jnp.mean(rewards),
        "reward_std": jnp.std(rewards),
        "reward_min": jnp.min(rewards),
        "reward_max": jnp.max(rewards),
        "steps_mean": jnp.mean(steps),
    }


# =============================================================================
# Data Collection for Decoder Training
# =============================================================================

def collect_data_for_decoder(
    agent: CGoRLAgent,
    env,
    prng: Array,
    num_envs: int,
    episode_length: int,
    num_iterations: int,
) -> tuple[Array, Array, Array]:
    """Collect (obs, action) pairs for decoder training.
    
    Returns:
        obs: Observations
        actions: Actual actions (not ε!)
        rewards: Episode rewards for filtering
    """
    rollout_state = CGoRLRolloutState.init(env, prng, num_envs)
    
    def step_fn(state, _):
        prng_sample, prng_next = jax.random.split(state.prng)
        
        # Get action through full pipeline
        action, eps, _ = agent.sample_action(
            state.env_state.obs, prng_sample, deterministic=False
        )
        
        env_action = jnp.tanh(action)
        next_env_state = jax.vmap(state.env.step)(state.env_state, env_action)
        
        # Auto-reset
        done = next_env_state.done.astype(bool)
        where_done = lambda x, y: jnp.where(
            done.reshape(done.shape + (1,) * (x.ndim - done.ndim)), x, y
        )
        next_env_state = next_env_state.replace(
            obs=jax.tree.map(where_done, state.first_obs, next_env_state.obs),
            data=jax.tree.map(where_done, state.first_data, next_env_state.data),
            done=jnp.zeros_like(next_env_state.done),
        )
        
        new_state = jdc.replace(state, env_state=next_env_state, prng=prng_next)
        
        return new_state, (state.env_state.obs, action, next_env_state.reward)
    
    final_state, (obs, actions, rewards) = jax.lax.scan(
        step_fn,
        rollout_state,
        length=num_iterations,
    )
    
    return obs, actions, rewards

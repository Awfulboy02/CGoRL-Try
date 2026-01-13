"""C-GoRL: Contrastive GoRL module."""

from .cgorl import (
    # Configuration
    CGoRLConfig,
    
    # CURL components
    CURLEncoderState,
    augment_state,
    compute_infonce_loss,
    
    # Latent policy
    LatentPolicyActionInfo,
    LatentPolicyTransition,
    LatentPolicyParams,
    
    # Main encoder state
    CGoRLEncoderState,
    
    # Agent
    CGoRLAgent,
    
    # Rollout
    CGoRLRolloutState,
    eval_cgorl_policy,
    collect_data_for_decoder,
)

__all__ = [
    "CGoRLConfig",
    "CURLEncoderState",
    "augment_state",
    "compute_infonce_loss",
    "LatentPolicyActionInfo",
    "LatentPolicyTransition", 
    "LatentPolicyParams",
    "CGoRLEncoderState",
    "CGoRLAgent",
    "CGoRLRolloutState",
    "eval_cgorl_policy",
    "collect_data_for_decoder",
]

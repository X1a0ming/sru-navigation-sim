# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom RSL-RL configuration classes for navigation tasks.

These config classes replace the standard Isaac Lab RL configs to support
custom network architectures used in navigation tasks with depth camera inputs.
"""

from dataclasses import MISSING
from typing import Literal, Optional

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks with navigation extensions."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    rnn_type: str = "lstm"
    """The type of RNN to use."""

    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    dropout: float = 0.0
    """The dropout rate for the first layer of the actor and critic networks."""

    # Visual inputs
    num_cameras: int = 1
    """Number of depth cameras encoded into the observation (1 or 2)."""
    
    image_input_dims: tuple[int, int, int] = (64, 5, 8)
    """Encoded depth feature shape as (C, H, W)."""
    
    height_input_dims: tuple[int, int, int] = (64, 7, 7)
    """Encoded height scan feature shape as (C, H, W). Default is (64, 7, 7) for 64*7*7=3136 features."""

    cache_num_slots: int = 16
    """Number of slots in Self-Cache policies."""

    cache_key_dim: int = 64
    """Self-Cache key dimension."""

    cache_value_dim: int = 64
    """Self-Cache value/readout dimension."""

    cache_gate_init: float = 0.02
    """Initial residual actor gate for Self-Cache policies."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = MISSING
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    value_clip_param: float = 0.2
    """The value clipping parameter. Default is 0.2."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    future_return_coef: float = 0.0
    """Coefficient for Self-Cache future-return auxiliary prediction."""

    future_return_horizons: list[int] = None
    """Future-return horizons used by Self-Cache auxiliary heads."""

    write_sparsity_coef: float = 0.0
    """Coefficient for Self-Cache write sparsity regularization."""

    cache_write_target: float = 0.1
    """Target average write gate for Self-Cache policies."""

    residual_action_l2_coef: float = 0.0
    """Coefficient for Self-Cache residual action L2 regularization."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: Optional[int] = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    reward_shifting_value: float = 0.0
    """The value to shift the reward by. Default is 0.0."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string."""

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all)."""

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all)."""

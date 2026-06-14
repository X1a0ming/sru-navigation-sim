# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RSL-RL agent configurations for B2W navigation tasks."""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class B2WNavMDPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MDPO runner configuration for B2W navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 500
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_b2w"
    experiment_name = "b2w_navigation_mdpo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),  # depth image: 64 channels * 5 * 8 = 2560
        height_input_dims=(64, 7, 7),  # encoded height_scan_critic: 64*7*7 = 3136
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MDPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="exponential",
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class B2WNavMDPORunnerDevCfg(B2WNavMDPORunnerCfg):
    """Development configuration for MDPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "b2w_navigation_mdpo_dev"
        self.logger = "tensorboard"


@configclass
class B2WNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for B2W navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 500
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_b2w"
    experiment_name = "b2w_navigation_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),  # depth image: 64 channels * 5 * 8 = 2560
        height_input_dims=(64, 7, 7),  # encoded height_scan_critic: 64*7*7 = 3136
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class B2WNavPPORunnerDevCfg(B2WNavPPORunnerCfg):
    """Development configuration for PPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "b2w_navigation_ppo_dev"
        self.logger = "tensorboard"


@configclass
class B2WNavPpoDeltaSRURunnerCfg(B2WNavPPORunnerCfg):
    """PPO runner configuration for B2W navigation with the Delta-SRU policy."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "b2w_navigation_deltasru_ppo"
        self.policy.class_name = "ActorCriticDeltaSRU"
        self.policy.rnn_type = "lstm_sru"


@configclass
class B2WNavPpoDeltaSRURunnerDevCfg(B2WNavPpoDeltaSRURunnerCfg):
    """Development configuration for B2W Delta-SRU PPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "b2w_navigation_deltasru_ppo_dev"
        self.logger = "tensorboard"


B2WNavDeltaSRUPPORunnerCfg = B2WNavPpoDeltaSRURunnerCfg
B2WNavDeltaSRUPPORunnerDevCfg = B2WNavPpoDeltaSRURunnerDevCfg


@configclass
class B2WNavPpoSelfCachedDeltaSRURunnerCfg(B2WNavPpoDeltaSRURunnerCfg):
    """PPO runner configuration for B2W navigation with Self-Cached Delta-SRU."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "b2w_navigation_self_cached_deltasru_ppo"
        self.policy.class_name = "ActorCriticSelfCachedDeltaSRU"
        self.policy.cache_num_slots = 16
        self.policy.cache_key_dim = 64
        self.policy.cache_value_dim = 64
        self.policy.cache_gate_init = 0.02
        self.algorithm.future_return_coef = 0.05
        self.algorithm.future_return_horizons = [8, 16, 32, 64]
        self.algorithm.write_sparsity_coef = 0.01
        self.algorithm.cache_write_target = 0.1
        self.algorithm.residual_action_l2_coef = 0.001


@configclass
class B2WNavPpoSelfCachedDeltaSRURunnerDevCfg(B2WNavPpoSelfCachedDeltaSRURunnerCfg):
    """Development configuration for B2W Self-Cached Delta-SRU PPO."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "b2w_navigation_self_cached_deltasru_ppo_dev"
        self.logger = "tensorboard"


B2WNavSelfCachedDeltaSRUPPORunnerCfg = B2WNavPpoSelfCachedDeltaSRURunnerCfg
B2WNavSelfCachedDeltaSRUPPORunnerDevCfg = B2WNavPpoSelfCachedDeltaSRURunnerDevCfg

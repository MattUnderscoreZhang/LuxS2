from gym import spaces
import numpy as np
from os import path
import sys
from luxai_s2.map_generator.generator import argparse
import torch
from torch import nn
from torch.functional import Tensor
from typing import Any, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.training.observations import get_minimap_obs


WEIGHTS_PATH = path.join(path.dirname(__file__), "logs/models/best_model.zip")
N_INPUTS = 56
N_MINIMAPS_PER_INPUT = 4
N_FEATURES = 64


class JobFeaturesNet(nn.Module):
    def __init__(self):
        n_in_channels = N_INPUTS * N_MINIMAPS_PER_INPUT
        self.per_pixel_branch = nn.Sequential(  # TODO: have branches operate identically on different minimap sizes
            nn.Conv2d(n_in_channels, 512, 1),
            nn.Tanh(),
            nn.Conv2d(512, 64, 1),
            nn.Tanh(),
        )
        # TODO: append original minimaps
        self.conv_layer = nn.Conv2d(N_INPUTS + 64, 16, 3, stride=3)  # TODO: conv identically on different minimap sizes
        self.fc_layer= nn.Linear(32 * 4 * 4, N_FEATURES)  # TODO: concat and flatten afterwards

    def forward(self, obs: Tensor) -> Tensor:
        x = torch.cat([
            obs,
            self.per_pixel_branch(obs),
        ], dim=1)
        x = torch.tanh(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = torch.tanh(x)
        return x  # batch_size x N_FEATURES

    def load_weights(self, state_dict: Any) -> None:
        # TODO: make weights load separately for each job type
        net_keys = [
            layer_name
            for layer in [
                "inception_1",
                "inception_3",
                "inception_5",
                "conv_reduction_layer",
                "skip_reduction_layer",
                "fc_layer",
            ]
            for layer_name in [
                f"features_extractor.net.{layer}.weight",
                f"features_extractor.net.{layer}.bias",
            ]
        ] + [
            layer_name
            for layer_name in state_dict.keys()
            if layer_name.startswith("mlp_extractor.")
        ] + [
            layer_name
            for layer_name in state_dict.keys()
            if layer_name.startswith("action_net.")
        ]
        loaded_state_dict = {}
        for sb3_key, model_key in zip(net_keys, self.state_dict().keys()):
            loaded_state_dict[model_key] = state_dict[sb3_key]
            print("loaded", sb3_key, "->", model_key, file=sys.stderr)
        self.load_state_dict(loaded_state_dict)


jobs = [
    "ice_miner",
    "ore_miner",
    "courier",
    "sabateur",
    "soldier",
    "factory",
]


class UnitsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.features_nets = {
            job: JobFeaturesNet()
            for job in jobs
        }
        for job, net in self.features_nets.items():
            self.add_module(job, net)

    def get_unit_jobs(self, unit_positions: np.ndarray) -> list[str]:
        return [
            "general"  # TODO: calculate unit jobs
            for pos in unit_positions
        ]

    def forward(self, full_obs: dict[str, np.ndarray]) -> Tensor:
        """
        Use net for each unit based on its job.
        """
        unit_positions = np.argwhere(full_obs["player_has_robot"][0])
        unit_jobs = self.get_unit_jobs(unit_positions)
        minimap_obs = [
            get_minimap_obs(full_obs, unit_position)
            for unit_position in unit_positions
        ]
        unit_features = torch.cat([
            self.features_nets[job](obs)
            for job, obs in zip(unit_jobs, minimap_obs)
        ], dim=1)
        return unit_features


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64
        self.policy_net = nn.Sequential(
            nn.Linear(N_FEATURES, self.latent_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(N_FEATURES, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.value_net(features)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCriticNet()


def get_model(env: SubprocVecEnv, args: argparse.Namespace) -> PPO:
    model = PPO(
        CustomActorCriticPolicy,
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs={
            "features_extractor_class": UnitsFeaturesExtractor,
            # "net_arch": [64],
        },
        verbose=1,
        n_epochs=2,
        target_kl=args.target_kl,
        gamma=args.gamma,
        tensorboard_log=args.log_path,
    )
    return model

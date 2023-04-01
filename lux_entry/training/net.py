import argparse
from gym import spaces
from os import path
import sys
import torch
from torch import nn, Tensor
from typing import Any, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.training.observations import get_minimap_obs


WEIGHTS_PATH = path.join(path.dirname(__file__), "logs/models/best_model.zip")
N_INPUTS = 56
N_MINIMAP_MAGNIFICATIONS = 4
N_FEATURES = 64


class JobFeaturesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_pixel_branch = nn.Sequential(
            nn.Conv2d(N_INPUTS, 32, 1),
            nn.Tanh(),
            nn.Conv2d(32, 8, 1),
            nn.Tanh(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.Tanh(),
        )
        self.fc_layer_1= nn.Linear(8 * N_MINIMAP_MAGNIFICATIONS * 12 * 12, 1024)
        self.fc_layer_2= nn.Linear(1024, N_FEATURES)

    def forward(self, minimap_obs: list[Tensor]) -> Tensor:
        x = torch.cat([
            self.per_pixel_branch(obs)
            for obs in minimap_obs
        ], dim=0)
        x = x.view(-1)
        x = self.fc_layer_1(x)
        x = torch.tanh(x)
        x = self.fc_layer_2(x)
        x = torch.tanh(x)
        assert x.shape == (N_FEATURES, )
        return x

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
    "generalist",
    "factory",
]


class UnitsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, N_FEATURES)
        self.features_nets = {
            job: JobFeaturesNet()
            for job in jobs
        }
        for job, net in self.features_nets.items():
            self.add_module(job, net)

    def get_unit_jobs(self, batch_unit_positions: list[Tensor]) -> list[list[str]]:
        return [
            [
                "generalist"  # TODO: calculate unit jobs
                for _ in unit_positions
            ]
            for unit_positions in batch_unit_positions
        ]

    def forward(self, batch_full_obs: dict[str, Tensor]) -> list[Tensor]:
        """
        Use net for each unit based on its job.
        """
        batch_robot_map = batch_full_obs["player_has_robot"][:,0]

        # TODO: remove after testing
        # add robots for testing
        batch_robot_map[0][3][6] = 1
        batch_robot_map[1][3][6] = 1
        batch_robot_map[1][5][2] = 1

        batch_unit_positions = [
            torch.argwhere(robot_map)
            for robot_map in batch_robot_map
        ]
        batch_unit_jobs = self.get_unit_jobs(batch_unit_positions)
        # batch_unit_jobs[batch_n][unit_n]: str
        batch_minimap_obs = [
            [
                get_minimap_obs(full_obs, unit_position)
                for unit_position in unit_positions
            ]
            for i, unit_positions in enumerate(batch_unit_positions)
            if (full_obs := {k: v[i] for k, v in batch_full_obs.items()})
        ]
        # batch_minimap_obs[batch_n][unit_n][obs_size]: Tensor (N_INPUTS x 12 x 12)
        batch_unit_features = [
            torch.cat([
                self.features_nets[job](obs).unsqueeze(0)
                for job, obs in zip(unit_jobs, unit_minimap_obs)
            ])
            if len(unit_jobs) > 0
            else torch.zeros(0, N_FEATURES)
            for unit_jobs, unit_minimap_obs in zip(batch_unit_jobs, batch_minimap_obs)
        ]
        # batch_unit_features[batch_n]: Tensor (n_units x N_FEATURES)
        return batch_unit_features


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

    def forward(self, batch_features: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        batch_policies = [
            self.forward_actor(features)
            for features in batch_features
        ]
        batch_values = [
            self.forward_critic(features)
            for features in batch_features
        ]
        return batch_policies, batch_values


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

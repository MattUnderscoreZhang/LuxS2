import argparse
from gym import spaces
import torch
from torch import nn, Tensor
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.lux.utils import add_batch_dimension
from lux_entry.training.observations import (
    get_minimap_obs, N_OBS_CHANNELS, N_MINIMAP_MAGNIFICATIONS
)


N_FEATURES = 64


class JobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_pixel_branch = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, 32, 1),
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
            self.per_pixel_branch(magnification_obs)
            if i == 0  # TODO: unfreeze other magnifications
            else torch.zeros(8, 12, 12)
            for i, magnification_obs in enumerate(minimap_obs)
        ], dim=0)
        x = x.view(-1)
        x = self.fc_layer_1(x)
        x = torch.tanh(x)
        x = self.fc_layer_2(x)
        x = torch.tanh(x)
        # assert x.shape == (N_FEATURES, )
        return x


jobs = [
    "ice_miner",
    "ore_miner",
    "courier",
    "sabateur",
    "soldier",
    "generalist",
    "factory",
]


MAX_ROBOTS = 256


class UnitsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, N_FEATURES)
        self.job_nets = {
            job: JobNet()
            for job in jobs
        }
        for job, net in self.job_nets.items():
            self.add_module(job, net)

    def get_robot_jobs(self, batch_robot_positions: list[Tensor]) -> list[list[str]]:
        return [
            [
                "generalist"  # TODO: calculate robot jobs
                for _ in robot_positions
            ]
            for robot_positions in batch_robot_positions
        ]

    def forward(self, batch_full_obs: dict[str, Tensor]) -> Tensor:
        """
        Use net to extract features for each robot based on its job.
        Output is padded to length MAX_ROBOTS.
        """
        # get robot positions
        batch_has_robot = batch_full_obs["player_has_robot"][:,0]
        batch_robot_positions = [
            torch.argwhere(robot_map)
            for robot_map in batch_has_robot
        ]

        # calculate robot jobs
        # type(batch_robot_jobs[batch_n][robot_n]): str
        batch_robot_jobs = self.get_robot_jobs(batch_robot_positions)

        # get minimap observations
        # batch_mini_obs[batch_n][robot_n][zoom_level].shape(): (N_OBS_CHANNELS, 12, 12)
        batch_mini_obs = [
            [
                get_minimap_obs(full_obs, robot_position)
                for robot_position in robot_positions
            ]
            for i, robot_positions in enumerate(batch_robot_positions)
            if (full_obs := {k: v[i] for k, v in batch_full_obs.items()})
        ]

        # calculate features for each robot using its job net
        # batch_robot_features[batch_n].shape(): (n_robots, N_FEATURES)
        batch_robot_features = [
            torch.cat([
                add_batch_dimension(self.job_nets[job](obs))
                for job, obs in zip(robot_jobs, robot_mini_obs)
            ])
            if len(robot_jobs) > 0
            else torch.zeros(0, N_FEATURES)
            for robot_jobs, robot_mini_obs in zip(batch_robot_jobs, batch_mini_obs)
        ]

        # perform masking
        # padded_robot_features.shape(): (batch_size, MAX_ROBOTS, N_FEATURES)
        batch_size = batch_full_obs["player_has_robot"].shape[0]
        padded_robot_features = torch.zeros(
            (batch_size, MAX_ROBOTS, N_FEATURES),
            dtype=torch.float32
        )
        for i, robot_features in enumerate(batch_robot_features):
            if len(robot_features) == 0:
                continue
            padded_robot_features[i, :len(robot_features)] = robot_features
        return padded_robot_features


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        LATENT_DIM = 64
        self.latent_dim_pi = 64 * MAX_ROBOTS
        self.latent_dim_vf = 64 * MAX_ROBOTS
        self.policy_net = nn.Sequential(
            nn.Linear(N_FEATURES, LATENT_DIM),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(N_FEATURES, LATENT_DIM),
            nn.ReLU(),
        )

    def forward_actor(self, batch_features: Tensor) -> Tensor:
        policy = self.policy_net(batch_features)
        return policy.view(policy.shape[0], -1)

    def forward_critic(self, batch_features: Tensor) -> Tensor:
        value = self.value_net(batch_features)
        return value.view(value.shape[0], -1)

    def forward(self, batch_features: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_actor(batch_features), self.forward_critic(batch_features)


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


"""
def load_weights(model: nn.Module, state_dict: Any) -> None:
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
    for sb3_key, model_key in zip(net_keys, model.state_dict().keys()):
        loaded_state_dict[model_key] = state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)
    model.load_state_dict(loaded_state_dict)
"""

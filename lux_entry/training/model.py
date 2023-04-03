import argparse
from gym import spaces
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.training.observations import N_OBS_CHANNELS


N_FEATURES = 128
MAX_ROBOTS = 256
N_MINIMAP_MAGNIFICATIONS = 4
N_JOBS = 8
N_ACTIONS = 12


class MapFeaturesExtractor(BaseFeaturesExtractor):
    """
    Upgrade per-pixel map information by calculating more features using surrounding pixels.
    Input is (batch_size, N_OBS_CHANNELS, 48, 48). Output is (batch_size, N_FEATURES, 48, 48).
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, N_FEATURES)
        n_channels = int(N_FEATURES / 4)
        self.inception_1 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 1)
        self.inception_3 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 3, padding=1)
        self.inception_5 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 5, padding=2)
        self.inception_7 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 7, padding=3)

    def forward(self, batch_full_obs: Tensor) -> Tensor:
        x = torch.cat([
            self.inception_1(batch_full_obs),
            self.inception_3(batch_full_obs),
            self.inception_5(batch_full_obs),
            self.inception_7(batch_full_obs),
        ], dim=1)
        return F.tanh(x)


class JobNet(nn.Module):
    """
    Figure out what job a unit at each location on the map should have.
    Returns a Tensor of shape (batch_size, N_JOBS, 48, 48), where dim 1 is softmax-normalized.
    """
    def __init__(self):
        super().__init__()
        self.map_reduction = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=3),
            nn.Tanh(),
            nn.Conv2d(64, 16, 4, stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 128, 7),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
        )
        self.job_finder = nn.Sequential(
            nn.Conv2d(N_FEATURES + 32, N_JOBS, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, batch_upgraded_obs: Tensor) -> Tensor:
        # calculate a feature vector that describes the whole map, and broadcast to each pixel
        x = self.map_reduction(batch_upgraded_obs)  # (batch_size, 32)
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 48, 48)  # (batch_size, 32, 48, 48))
        x = torch.cat([batch_upgraded_obs, x], dim=1)  # (batch_size, N_FEATURES + 32, 48, 48)
        # calculate job logits per map pixel
        return self.job_finder(x)


class JobActionNet(nn.Module):
    """
    Make action decision based on information in surrounding 11x11 map grid.
    Returns shape (batch_size, N_ACTIONS, 48, 48), where dim 1 is softmax-normalized.
    """
    def __init__(self):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Tanh(),
            nn.Conv2d(64, 8, 1),
            nn.Tanh(),
            nn.ConstantPad2d(padding=5, value=-1),
            nn.Conv2d(8, 1024, 11),
            nn.Tanh(),
            nn.Conv2d(1024, N_ACTIONS, 1),
            nn.Tanh(),
        )

    def forward(self, batch_upgraded_obs: Tensor) -> Tensor:
        return self.action_net(batch_upgraded_obs)


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        PI_DIM = 64
        VF_DIM = 64
        N_ACTIONS = 12
        self.latent_dim_pi = PI_DIM * MAX_ROBOTS
        self.latent_dim_vf = VF_DIM * MAX_ROBOTS
        self.policy_net = nn.Sequential(
            nn.Linear(N_FEATURES, PI_DIM),
            nn.ReLU(),
            nn.Linear(PI_DIM, N_ACTIONS),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(N_FEATURES, VF_DIM),
            nn.ReLU(),
            nn.Linear(VF_DIM, 1),
            nn.ReLU(),
        )
        self.job_net = JobNet()
        self.job_action_nets = {
            job: JobActionNet()
            for job in [
                "ice_miner",
                "ore_miner",
                "courier",
                "sabateur",
                "soldier",
                "berserker",
                "generalist",
                "factory",
            ]
        }
        for job, net in self.job_action_nets.items():
            self.add_module(job, net)

    def forward_actor(self, batch_features: Tensor) -> Tensor:
        policy = self.policy_net(batch_features)
        return policy.reshape(policy.shape[0], -1)

    def forward_critic(self, batch_features: Tensor) -> Tensor:
        value = self.value_net(batch_features)
        return value.mean(dim=1).squeeze(-1)

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
        self.mlp_extractor = ActorCriticNet()
        self.action_net = nn.Identity()  # no Linear layer
        self.value_net = nn.Identity()  # no Linear layer


def get_model(env: SubprocVecEnv, args: argparse.Namespace) -> PPO:
    model = PPO(
        CustomActorCriticPolicy,
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs={
            "features_extractor_class": MapFeaturesExtractor,
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

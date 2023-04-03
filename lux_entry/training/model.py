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
N_MINIMAP_MAGNIFICATIONS = 4
JOBS = [
    "ice_miner", "ore_miner", "courier", "sabateur",
    "soldier", "berserker", "generalist", "factory"
]
N_ACTIONS = 12


class MapFeaturesExtractor(BaseFeaturesExtractor):
    """
    Upgrade per-pixel map information by calculating more features using surrounding pixels.
    Input is (batch_size, N_OBS_CHANNELS, 48, 48). Output is (batch_size, N_FEATURES * 48 * 48).
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, N_FEATURES * 48 * 48)
        n_channels = int(N_FEATURES / 4)
        self.inception_1 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 1)
        self.inception_3 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 3, padding=1)
        self.inception_5 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 5, padding=2)
        self.inception_7 = nn.Conv2d(N_OBS_CHANNELS, n_channels, 7, padding=3)

    def forward(self, batch_full_obs: Tensor) -> Tensor:
        batch_full_obs = torch.cat([v for v in batch_full_obs.values()], dim=1)
        x = torch.cat([
            self.inception_1(batch_full_obs),
            self.inception_3(batch_full_obs),
            self.inception_5(batch_full_obs),
            self.inception_7(batch_full_obs),
        ], dim=1)
        x = F.tanh(x)
        # return x.view(x.shape[0], -1)
        return x


class JobNet(nn.Module):
    """
    Figure out what job a unit at each location on the map should have.
    Returns Tensor of shape (batch_size, len(JOBS), 48, 48), where dim 1 is softmax-normalized.
    """
    def __init__(self):
        super().__init__()
        self.map_reduction = nn.Sequential(
            nn.Conv2d(N_FEATURES, 64, 3, stride=3),
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
        self.job_probs = nn.Sequential(
            nn.Conv2d(N_FEATURES + 32, len(JOBS), 1),
            nn.Softmax(dim=1),
        )

    def forward(self, batch_map_features: Tensor) -> Tensor:
        # calculate a feature vector that describes the whole map, and broadcast to each pixel
        x = self.map_reduction(batch_map_features)  # (batch_size, 32)
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 48, 48)  # (batch_size, 32, 48, 48)
        x = torch.cat([batch_map_features, x], dim=1)  # (batch_size, N_FEATURES + 32, 48, 48)
        return self.job_probs(x)


class JobActionNet(nn.Module):
    """
    Make action decision based on information in surrounding 11x11 map grid.
    Returns shape (batch_size, N_ACTIONS, 48, 48), where dim 1 is softmax-normalized.
    """
    def __init__(self):
        super().__init__()
        self.action_probs = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Tanh(),
            nn.Conv2d(64, 8, 1),
            nn.Tanh(),
            nn.ConstantPad2d(padding=5, value=-1),
            nn.Conv2d(8, 1024, 11),
            nn.Tanh(),
            nn.Conv2d(1024, N_ACTIONS, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, batch_map_features: Tensor) -> Tensor:
        return self.action_probs(batch_map_features)


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = N_ACTIONS * 48 * 48
        self.latent_dim_vf = 1
        self.job_net = JobNet()
        self.job_action_nets = {job: JobActionNet() for job in JOBS}
        for job, net in self.job_action_nets.items():
            self.add_module(job, net)
        self.value_layer = nn.Linear(32, 1)

    def forward_actor(self, batch_map_features: Tensor) -> Tensor:
        batch_map_features = batch_map_features.view(batch_map_features.shape[0], -1, 48, 48)
        job_probs = self.job_net(batch_map_features)
        job_action_probs = torch.stack([
            self.job_action_nets[job](batch_map_features) * job_probs[:, i].unsqueeze(1)
            for i, job in enumerate(JOBS)
        ], dim=1)
        job_action_probs = job_action_probs.sum(dim=1)
        return job_action_probs.reshape(job_action_probs.shape[0], -1)

    def forward_critic(self, batch_map_features: Tensor) -> Tensor:
        batch_map_features = batch_map_features.view(batch_map_features.shape[0], -1, 48, 48)
        batch_map_features = self.job_net.map_reduction(batch_map_features)
        return self.value_layer(batch_map_features)

    def forward(self, batch_map_features: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_actor(batch_map_features), self.forward_critic(batch_map_features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.mlp_extractor = ActorCriticNet()
        self.action_net = nn.Identity()  # no additional Linear layer
        self.value_net = nn.Identity()  # no additional Linear layer


def get_model(env: SubprocVecEnv, args: argparse.Namespace) -> PPO:
    model = PPO(
        CustomActorCriticPolicy,
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs={"features_extractor_class": MapFeaturesExtractor},
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

import argparse
import gym
from gym import spaces
from os import path
import sys
import torch
from torch import nn
from torch.functional import Tensor
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from lux_entry.components.types import PolicyNet


WEIGHTS_PATH = path.join(path.dirname(__file__), "logs/models/best_model.zip")


N_CONV_OBS = 26
N_SKIP_OBS = 1
N_FEATURES = 128
N_ACTIONS = 12


class Net(PolicyNet):
    def __init__(self):
        """
        This net is used during both training and evaluation.
        Net creation needs to take no arguments.
        The net contains both a feature extractor and a fully-connected policy layer.
        """
        super().__init__()
        self.n_actions = N_ACTIONS
        self.n_features = N_FEATURES
        self.reduction_layer_1 = nn.Conv2d(N_CONV_OBS * 4, 30, 1)
        self.tanh_layer_1 = nn.Tanh()
        self.conv_layer_1 = nn.Conv2d(30, 15, 3, padding='same')
        self.conv_layer_2 = nn.Conv2d(30, 15, 5, padding='same')
        self.tanh_layer_2 = nn.Tanh()
        self.reduction_layer_2 = nn.Conv2d(30, 5, 1)
        self.tanh_layer_3 = nn.Tanh()
        self.fc_layer= nn.Linear((5 + N_SKIP_OBS * 4) * 12 * 12, self.n_features)
        self.tanh_layer_4 = nn.Tanh()
        self.action_layer_1 = nn.Linear(self.n_features, 64)
        self.action_layer_2 = nn.Linear(64, self.n_actions)

    def extract_features(self, obs: Dict[str, Tensor]) -> Tensor:
        x = self.reduction_layer_1(obs["conv_obs"])
        x = self.tanh_layer_1(x)
        x = torch.cat([self.conv_layer_1(x), self.conv_layer_2(x)], dim=1)
        x = self.tanh_layer_2(x)
        x = self.reduction_layer_2(x)
        x = self.tanh_layer_3(x)
        x = torch.cat([x, obs["skip_obs"]], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = self.tanh_layer_4(x)
        return x

    def act(
        self, x: Dict[str, Tensor], action_masks: Tensor, deterministic: bool = False
    ) -> Tensor:
        features = self.extract_features(x)
        x = self.action_layer_1(features)
        x = nn.Tanh()(x)
        action_logits = self.action_layer_2(x)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()

    def load_weights(self, state_dict: Any) -> None:
        net_keys = [
            layer_name
            for layer in [
                "reduction_layer_1",
                "conv_layer_1",
                "conv_layer_2",
                "reduction_layer_2",
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


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        """
        This class is only used by the model function below during training.
        The Net forward function has a fully-connected net after the feature extractor.
        We call only the feature extractor, and SB3 adds (a) fully-connected layer(s) afterwards.
        """
        super().__init__(observation_space, N_FEATURES)
        self.net = Net()

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        return self.net.extract_features(obs)


def model(env: gym.Env, args: argparse.Namespace):
    """
    This model is only used for training.
    SB3 adds a fully-connected net after the feature extractor.
    Fully-connected hidden-layer shapes can be manually specified via the net_arch parameter.
    """
    return PPO(
        "MultiInputPolicy",
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs={
            "features_extractor_class": CustomFeatureExtractor,
            "net_arch": [64],
        },
        verbose=1,
        n_epochs=2,
        target_kl=args.target_kl,
        gamma=args.gamma,
        tensorboard_log=path.join(args.log_path),
    )

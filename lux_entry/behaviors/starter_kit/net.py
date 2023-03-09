import argparse
import gym
from gym import spaces
from os import path
import sys
import torch
from torch import nn
from torch.functional import Tensor
from typing import Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from lux_entry.components.types import PolicyNet


WEIGHTS_PATH = path.join(path.dirname(__file__), "logs/models/best_model.zip")


N_OBSERVABLES = 13
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
        self.n_features = N_FEATURES
        self.n_actions = N_ACTIONS
        self.features_net = nn.Sequential(
            nn.Linear(N_OBSERVABLES, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_features),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(self.n_features, self.n_actions)

    def evaluate(
        self,
        x: Tensor,
        action_masks: Optional[Tensor] = None,
        deterministic: bool = False
    ) -> Tensor:
        x = self.features_net(x)
        action_logits = self.action_net(x)
        if action_masks is not None:
            action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()

    def load_weights(self, state_dict: Any) -> None:
        net_keys = [
            key
            for key in state_dict.keys()
            if key.startswith("features_extractor.net.features_net")
        ]
        net_keys += [
            key
            for key in state_dict.keys()
            if key.startswith("action_net.")
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

    def forward(self, obs: Tensor) -> Tensor:
        return self.net.features_net(obs)


def model(env: gym.Env, args: argparse.Namespace):
    """
    This model is only used for training.
    SB3 adds a fully-connected net after the feature extractor.
    Fully-connected hidden-layer shapes can be manually specified via the net_arch parameter.
    """
    return PPO(
        "MlpPolicy",
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs={
            "features_extractor_class": CustomFeatureExtractor,
            "net_arch": [],
            # "net_arch": [128, 128],
        },
        verbose=1,
        n_epochs=2,
        target_kl=args.target_kl,
        gamma=args.gamma,
        tensorboard_log=path.join(args.log_path),
    )

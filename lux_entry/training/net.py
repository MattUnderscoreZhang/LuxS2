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

from luxai_s2.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player
from lux_entry.lux.utils import add_batch_dimension
from lux_entry.training.env import ObservationWrapper


WEIGHTS_PATH = path.join(path.dirname(__file__), "logs/models/best_model.zip")


N_CONV_OBS = 26
N_SKIP_OBS = 1
N_FEATURES = 128
N_ACTIONS = 12


# TODO: if minimap extraction is done here, it call all be done in one batch
"""
def _mean_pool(arr: Tensor, window: int) -> Tensor:
    return F.avg_pool2d(arr.unsqueeze(0), window, stride=window).squeeze(0)

def _get_minimaps(full_obs: MapFeaturesObservation, x: int, y: int) -> dict[str, Tensor]:
    \"""
    Create minimaps for a set of features around (x, y).
    \"""
    # observables to get minimaps for, as (observable, skip_obs)
    minimap_obs = [
        (full_obs.tile_has_ice, True),
        (full_obs.tile_per_player_has_factory, False),
        (full_obs.tile_per_player_has_robot, False),
        (full_obs.tile_per_player_has_light_robot, False),
        (full_obs.tile_per_player_has_heavy_robot, False),
        (full_obs.tile_rubble, False),
        (full_obs.tile_per_player_light_robot_power, False),
        (full_obs.tile_per_player_heavy_robot_power, False),
        (full_obs.tile_per_player_factory_ice_unbounded, False),
        (full_obs.tile_per_player_factory_ore_unbounded, False),
        (full_obs.tile_per_player_factory_water_unbounded, False),
        (full_obs.tile_per_player_factory_metal_unbounded, False),
        (full_obs.tile_per_player_factory_power_unbounded, False),
        (full_obs.game_is_day, False),
        (full_obs.game_day_or_night_elapsed, False),
    ]

    # create minimaps centered around x, y
    def get_expanded_map(full_map_obs: np.ndarray) -> Tensor:
        expanded_map = torch.full((full_map_obs.shape[0], 96, 96), -1.0)
        # unit is in lower right pixel of upper left quadrant
        expanded_map[:, x:x+48, y:y+48] = Tensor(full_map_obs)
        return expanded_map

    expanded_maps = torch.cat([
        get_expanded_map(full_map_obs)
        for full_map_obs, _ in minimap_obs
    ], dim=0)
    conv_minimaps = torch.cat([
        # small map (12x12 area)
        expanded_maps[:, 42:54, 42:54],
        # medium map (24x24 area)
        _mean_pool(expanded_maps[:, 36:60, 36:60], 2),
        # large map (48x48 area)
        _mean_pool(expanded_maps[:, 24:72, 24:72], 4),
        # full map (96x96 area)
        _mean_pool(expanded_maps, 8),
    ], dim=0)
    is_skip_dim = [
        skip
        for full_map_obs, skip in minimap_obs
        for _ in range(full_map_obs.shape[0])
    ] * 4
    skip_minimaps = conv_minimaps[is_skip_dim]

    return {"conv_obs": conv_minimaps, "skip_obs": skip_minimaps}
"""


class Net(nn.Module):
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

    def extract_features(self, obs: dict[str, Tensor]) -> Tensor:
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

    def evaluate(
        self,
        x: dict[str, Tensor],
        action_masks: Optional[Tensor] = None,
        deterministic: bool = False
    ) -> Tensor:
        features = self.extract_features(x)
        x = self.action_layer_1(features)
        x = nn.Tanh()(x)
        action_logits = self.action_layer_2(x)
        if action_masks is not None:
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

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
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

from gym import spaces
from os import path
import sys
from luxai_s2.map_generator.generator import argparse
import torch
from torch import nn
from torch.functional import Tensor
from typing import Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv


WEIGHTS_PATH = path.join(path.dirname(__file__), "logs/models/best_model.zip")


N_CONV_OBS = 26
N_SKIP_OBS = 1
N_FEATURES = 128
N_ACTIONS = 12


class JobNet(nn.Module):
    def __init__(self):
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
        self.tanh_layer_5 = nn.Tanh()
        self.action_layer_2 = nn.Linear(64, self.n_actions)

    def forward(
        self,
        obs: dict[str, Tensor],
        action_masks: Optional[Tensor] = None,
        deterministic: bool = False
    ) -> Tensor:
        conv_obs, skip_obs = obs["conv_obs"], obs["skip_obs"]
        x = self.reduction_layer_1(conv_obs)
        x = self.tanh_layer_1(x)
        x = torch.cat([self.conv_layer_1(x), self.conv_layer_2(x)], dim=1)
        x = self.tanh_layer_2(x)
        x = self.reduction_layer_2(x)
        x = self.tanh_layer_3(x)
        x = torch.cat([x, skip_obs], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = self.tanh_layer_4(x)
        x = self.action_layer_1(x)
        x = self.tanh_layer_5(x)
        action_logits = self.action_layer_2(x)
        if action_masks is not None:
            action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()


class UnitsNet(nn.Module):
    def __init__(self):
        """
        This net is used during both training and evaluation.
        Net creation needs to take no arguments.
        The net contains both a feature extractor and a fully-connected policy layer.
        """
        super().__init__()
        jobs = [
            "ice_miner",
            "ore_miner",
            "courier",
            "sabateur",
            "scout",
            "soldier",
            "builder",
            "factory",
        ]
        self.nets = {
            job: JobNet()
            for job in jobs
        }
        for job, net in self.nets.items():
            self.add_module(job, net)

        self.latent_dim_pi = 64
        self.latent_dim_vf = 64
        self.policy_net = nn.Sequential(
            nn.Linear(N_FEATURES * len(jobs), self.latent_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(N_FEATURES * len(jobs), self.latent_dim_pi),
            nn.ReLU(),
        )

    def extract_features(
        self,
        x: dict[str, Tensor],
        action_masks: Optional[Tensor] = None,
        deterministic: bool = False
    ) -> Tensor:
        unit_features = {
            unit_id: self.nets[job].extract_features(conv_obs, skip_obs)
            for unit_id, unit_obs in obs.items()
            if (job := str(unit_obs["job"]))
            and (conv_obs := Tensor(unit_obs["conv_obs"]))
            and (skip_obs := Tensor(unit_obs["skip_obs"]))
        }
        x = self.action_layer_1(unit_features)
        x = nn.Tanh()(x)
        action_logits = self.action_layer_2(x)
        if action_masks is not None:
            action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()

    def forward_actor(self, obs: Tensor) -> Tensor:
        features = self.extract_features(obs)
        return self.policy_net(features)

    def forward_critic(self, obs: Tensor) -> Tensor:
        features = self.extract_features(obs)
        return self.value_net(features)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward_actor(obs), self.forward_critic(obs)

    def load_weights(self, state_dict: Any) -> None:
        # TODO: make weights load separately for each job type
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
        self.mlp_extractor = UnitsNet()


def get_model(env: SubprocVecEnv, args: argparse.Namespace) -> PPO:
    model = PPO(
        CustomActorCriticPolicy,
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=1,
        n_epochs=2,
        target_kl=args.target_kl,
        gamma=args.gamma,
        tensorboard_log=args.log_path,
    )
    # model = PPO(
        # "MultiInputPolicy",
        # env,
        # n_steps=args.rollout_steps // args.n_envs,
        # batch_size=args.batch_size,
        # learning_rate=args.learning_rate,
        # policy_kwargs={
            # "features_extractor_class": CustomFeatureExtractor,
            # "net_arch": [64],
        # },
        # verbose=1,
        # n_epochs=2,
        # target_kl=args.target_kl,
        # gamma=args.gamma,
        # tensorboard_log=args.log_path,
    # )
    return model

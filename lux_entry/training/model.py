import argparse
from gym import spaces
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from typing import Callable, Dict, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.training.observations import N_OBS_CHANNELS


N_MAP_FEATURES = 128
N_MINIMAP_MAGNIFICATIONS = 4
ROBOT_JOBS = ["ice_miner", "ore_miner", "courier", "sabateur", "soldier", "berserker"]
N_ACTIONS = 12


# TODO: this code is CPU-bound - figure out how to optimize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MapFeaturesExtractor(BaseFeaturesExtractor):
    """
    Upgrade per-pixel map information by calculating more features using surrounding pixels.
    Input is (batch_size, N_OBS_CHANNELS, 48, 48).
    Output is (batch_size, N_MAP_FEATURES * 48 * 48).
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, N_MAP_FEATURES * 48 * 48)
        N_LAYER_FEATURES = 32
        self.single_pixel_features = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, N_LAYER_FEATURES, 1),
            nn.Tanh(),
        ).to(device)
        self.whole_map_features = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, 16, 3, stride=3),
            nn.Tanh(),
            nn.Conv2d(16, 16, 4, stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 32, 7),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(32, N_MAP_FEATURES - N_OBS_CHANNELS - N_LAYER_FEATURES),  # * 2),
            nn.Tanh(),
        ).to(device)

    def forward(self, batch_full_obs: Dict[str, Tensor]) -> Tensor:
        # place my units as the first channels, to use for masking later
        my_factories = batch_full_obs["player_has_factory"][:, 0].unsqueeze(1)
        opp_factories = batch_full_obs["player_has_factory"][:, 1].unsqueeze(1)
        my_robots = batch_full_obs["player_has_robot"][:, 0].unsqueeze(1)
        opp_robots = batch_full_obs["player_has_robot"][:, 1].unsqueeze(1)
        batch_full_obs.pop("player_has_factory")
        batch_full_obs.pop("player_has_robot")
        x = torch.cat(
            [my_factories, my_robots, opp_factories, opp_robots] +
            [v for v in batch_full_obs.values()], dim=1
        ).to(device)
        # calculate a feature vector that describes the whole map, and broadcast to each pixel
        """
        whole_map_features = (
            self.whole_map_features(x)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 48, 48)
        )
        """
        # TODO: put masks on what map feature sets to calculate and use
        x = torch.cat([
            x,
            torch.zeros(x.shape[0], 32, 48, 48).to(device),
            # self.single_pixel_features(x),
            torch.zeros(x.shape[0], N_MAP_FEATURES - N_OBS_CHANNELS - 32, 48, 48).to(device),
            # whole_map_features,
        ], dim=1)
        return x


class JobNet(nn.Module):
    """
    Figure out what job a unit at each location on the map should have.
    Returns Tensor of shape (batch_size, len(ROBOT_JOBS), 48, 48). dim 1 is softmax-normalized.
    """
    def __init__(self):
        super().__init__()
        self.job_probs = nn.Sequential(
            nn.Conv2d(N_MAP_FEATURES, 32, 1),
            nn.Tanh(),
            nn.Conv2d(32, 8, 1),
            nn.Tanh(),
            nn.Conv2d(8, len(ROBOT_JOBS), 9, padding=4),
            nn.Softmax(dim=1),
        ).to(device)

    def forward(self, batch_map_features: Tensor) -> Tensor:
        return self.job_probs(batch_map_features)


class JobActionNet(nn.Module):
    """
    Make per-pixel action decisions. Pixels features already contain info about surroundings.
    Returns shape (batch_size, N_ACTIONS, 48, 48), where dim 1 is normalized.
    """
    def __init__(self):
        super().__init__()
        self.nearby_features = nn.Sequential(
            nn.Conv2d(N_MAP_FEATURES, 32, 1),
            nn.Tanh(),
            nn.Conv2d(32, 16, 1),
            nn.Tanh(),
            nn.Conv2d(16, 8, 5, padding=2),
        ).to(device)
        self.medium_features = nn.Sequential(
            nn.Conv2d(N_MAP_FEATURES, 16, 1),
            nn.Tanh(),
            nn.Conv2d(16, 8, 1),
            nn.Tanh(),
            nn.Conv2d(8, 4, 9, padding=4),
        ).to(device)
        self.distant_features = nn.Sequential(
            nn.Conv2d(N_MAP_FEATURES, 8, 1),
            nn.Tanh(),
            nn.Conv2d(8, 2, 1),
            nn.Tanh(),
            nn.Conv2d(2, 4, 11, padding=5),
        ).to(device)
        self.action_logits = nn.Conv2d(16, N_ACTIONS, 1).to(device)

    def forward(self, batch_map_features: Tensor) -> Tensor:
        all_features = torch.cat([
            self.nearby_features(batch_map_features),
            self.medium_features(batch_map_features),
            self.distant_features(batch_map_features),
        ], dim=1)
        action_logits = self.action_logits(all_features)
        normalization = action_logits.sum(dim=1, keepdim=True).clamp(min=1e-6)
        norm_action_logits = action_logits / normalization
        return norm_action_logits


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.job_net = JobNet()
        self.job_action_nets = [JobActionNet() for _ in ROBOT_JOBS + ["factory"]]
        for job, net in zip(ROBOT_JOBS + ["factory"], self.job_action_nets):
            self.add_module(job, net)
        self.set_active_robot_jobs(["ice_miner"])
        self.value_calculation = nn.Sequential(
            nn.Conv2d(N_MAP_FEATURES, 32, 2, stride=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, 2, stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 8, 4, stride=4),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(72, 1),
        ).to(device)

    def set_active_robot_jobs(self, active_nets: List[str]):
        self.robot_job_active = [job in active_nets for job in ROBOT_JOBS]

    def forward_actor(self, batch_map_features: Tensor) -> Tensor:
        batch_map_features = batch_map_features.view(batch_map_features.shape[0], -1, 48, 48)
        # get units - robots take precedence in action calculations
        my_factories = batch_map_features[:, 0].unsqueeze(1)
        my_robots = batch_map_features[:, 1].unsqueeze(1)
        my_factories = my_factories * (1 - my_robots)
        # find best job and action for each robot
        """
        robot_job_probs = self.job_net(batch_map_features)
        """
        action_map_size = torch.Size([
            batch_map_features.shape[0],
            N_ACTIONS,
            *batch_map_features.shape[2:],
        ])
        robot_action_logits = self.job_action_nets[0](batch_map_features)
        """
        robot_action_logits = torch.stack([
            self.job_action_nets[i](batch_map_features)
            if active
            else torch.zeros(action_map_size)
            for i, active in enumerate(self.robot_job_active)
        ], dim=1)
        robot_action_logits = robot_action_logits * robot_job_probs.unsqueeze(2)
        robot_action_logits = robot_action_logits.sum(dim=1)
        """
        # find best action for each factory
        factory_action_logits = self.job_action_nets[-1](batch_map_features)
        # return action logits
        action_logits = (
            robot_action_logits * my_robots +
            factory_action_logits * my_factories
        )
        # TODO: normalize action logits correctly
        # TODO: figure out how to get only unmasked action logits to matter
        return action_logits

    def forward_critic(self, batch_map_features: Tensor) -> Tensor:
        return self.value_calculation(batch_map_features)

    def forward(self, batch_map_features: Tensor) -> Tuple[Tensor, Tensor]:
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

    def _get_action_dist_from_latent(self, latent_pi: Tensor) -> Distribution:
        # much faster than default Distribution
        class MyDistribution(torch.distributions.Categorical):
            def get_actions(self, deterministic: bool = False) -> Tensor:
                params: Tensor = self._param
                actions = (
                    torch.argmax(params, dim=-1)
                    if deterministic
                    else super().sample()
                )
                return actions.view(params.shape[0], -1)

            def log_prob(self, actions: Tensor) -> Tensor:
                params: Tensor = self._param
                actions = actions.view(params.shape[0], 48, 48).long()
                log_probs = params.log_softmax(dim=-1)
                action_log_probs = (
                    log_probs.gather(-1, actions.unsqueeze(-1))
                    .squeeze(-1).mean(dim=(1,2))  # using mean to avoid huge KL div
                )
                return action_log_probs

        action_logits = latent_pi.permute(0, 2, 3, 1).contiguous()
        return MyDistribution(logits=action_logits)


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

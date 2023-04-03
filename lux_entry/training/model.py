import argparse
from gym import spaces
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.lux.utils import add_batch_dimension
from lux_entry.training.observations import N_OBS_CHANNELS


N_FEATURES = 64
MAX_ROBOTS = 256
N_MINIMAP_MAGNIFICATIONS = 4
N_JOBS = 8


def get_mini_obs(
    batch_full_obs: dict[str, Tensor], batch_pos: list[list[Tensor]]
) -> list[list[Tensor]]:
    """
    Create minimaps for a set of features around (x, y).
    Return a list of four minimap magnifications, each with all features concated.
    """
    def _mean_pool(arr: Tensor, scale: int) -> Tensor:
        # return F.avg_pool2d(arr, kernel_size=scale, stride=scale)
        return F.interpolate(arr, scale_factor=scale, mode='bilinear', align_corners=True)

    def _get_minimaps(
        expanded_map: Tensor,
        batch_pos: list[list[Tensor]]
    ) -> list[list[Tensor]]:
        minimaps = [
            [
                expanded_map[:, x-6:x+6, y-6:y+6],  # small map (12x12)
                _mean_pool(expanded_map[:, x-12:x+12, y-12:y+12], 2),  # medium map (24x24)
                _mean_pool(expanded_map[:, x-24:x+24, y-24:y+24], 4),  # large map (48x48)
                _mean_pool(expanded_map[:, x-48:x+48, y-48:y+48], 8),  # full map (96x96)
            ]
            for pos in batch_pos
            if (x := pos[0] + 48) and (y := pos[1] + 48)
        ]
        return minimaps

    mini_obs = _get_minimaps(expanded_map, batch_pos)
    return mini_obs


class JobNet(nn.Module):
    """
    Figure out what job a unit at each location on the map should have.
    Returns a Tensor of shape (batch_size, N_JOBS, 48, 48), where dim 1 is softmax-normalized.
    """
    def __init__(self):
        super().__init__()
        self.inception_1 = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, 32, 1),
            nn.Tanh(),
        )
        self.inception_3 = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, 32, 3, padding=1),
            nn.Tanh(),
        )
        self.inception_5 = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, 32, 5, padding=2),
            nn.Tanh(),
        )
        self.inception_7 = nn.Sequential(
            nn.Conv2d(N_OBS_CHANNELS, 32, 7, padding=3),
            nn.Tanh(),
        )
        self.map_reduction = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=3),
            nn.Tanh(),
            nn.Conv2d(64, 16, 4, stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 128, 7),
            nn.Tanh(),
            nn.Flatten(),
        )
        self.fc_layer_1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.fc_layer_2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.Tanh(),
        )
        self.role_finder = nn.Sequential(
            nn.Conv2d(160, N_JOBS, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, batch_full_obs: Tensor) -> Tensor:
        # for each map pixel, calculate features based on surrounding pixels
        # batch_full_obs: (batch_size, N_OBS_CHANNELS, 48, 48)
        x = torch.cat([
            self.inception_1(batch_full_obs),
            self.inception_3(batch_full_obs),
            self.inception_5(batch_full_obs),
            self.inception_7(batch_full_obs),
        ], dim=1)
        x = torch.tanh(x)  # (batch_size, 128, 48, 48)
        batch_upgraded_full_obs = x
        # calculate a feature vector that describes the whole map
        x = self.map_reduction(x)  # (batch_size, 128)
        x = self.fc_layer_1(x)  # (batch_size, 128)
        x = self.fc_layer_2(x)  # (batch_size, 32)
        # add feature vector to each map pixel
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(-1, -1, 48, 48)  # (batch_size, 32, 48, 48))
        x = torch.cat([batch_upgraded_full_obs, x], dim=1)  # (batch_size, 160, 48, 48)
        # calculate role logits per map pixel
        x = self.role_finder(x)
        return x


class ActionNet(nn.Module):
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
        return x


class UnitsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, N_FEATURES)
        self.job_nets = {
            job: ActionNet()
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
        for job, net in self.job_nets.items():
            self.add_module(job, net)

    def forward(self, batch_full_obs: dict[str, Tensor]) -> Tensor:
        """
        Use net to extract features for each robot based on its job.
        Output is padded to length MAX_ROBOTS.
        """
        # get robot positions
        # sort by x position, then y position
        batch_has_robot = batch_full_obs["player_has_robot"][:,0]
        batch_robot_positions = [
            sorted(torch.argwhere(robot_map), key=lambda x: (x[0], x[1]))
            for robot_map in batch_has_robot
        ]

        # calculate robot jobs
        # type(batch_robot_jobs[batch_n][robot_n]): str
        # batch_robot_jobs = self.get_robot_jobs(batch_robot_positions)

        # get padded full map Tensor
        # batch_expanded_obs.shape(): (batch_n, N_OBS_CHANNELS, 144, 144)
        stacked_full_obs = torch.cat(
            [
                batch_full_obs[key]
                for key in batch_full_obs.keys()
            ], dim=1
        )
        # batch_expanded_obs = F.pad(stacked_full_obs, (48, 48, 48, 48), value=-1)

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

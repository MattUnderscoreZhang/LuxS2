import argparse
import os.path as osp
import torch
from torch.functional import Tensor
import torch.nn as nn
from typing import Any

from stable_baselines3 import PPO


this_directory = osp.dirname(__file__)
WEIGHTS_PATH = osp.join(this_directory, "logs/models/best_model.zip")


class Net(nn.Module):
    def __init__(self, len_output: int = 12):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, len_output),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    def act(
        self, x: Tensor, action_masks: Tensor, deterministic: bool = False
    ) -> Tensor:
        action_logits = self.forward(x)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()


def model(env: Any, args: argparse.Namespace):
    return PPO(
        "MlpPolicy",
        env,
        n_steps=args.rollout_steps // args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs=dict(net_arch=(128, 128)),
        verbose=1,
        n_epochs=2,
        target_kl=args.target_kl,
        gamma=args.gamma,
        tensorboard_log=osp.join(args.log_path),
    )

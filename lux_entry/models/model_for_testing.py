import torch
from torch.functional import Tensor
import torch.nn as nn


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

    def act(self, x: Tensor, action_masks: Tensor, deterministic: bool = False) -> Tensor:
        action_logits = self.forward(x)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()

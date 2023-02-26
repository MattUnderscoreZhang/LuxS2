import torch
from torch import nn
from torch.functional import Tensor


class MlpNet(nn.Module):
    def __init__(self, n_observables: int, n_features: int, n_actions: int):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(n_observables, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_features),
            nn.Tanh(),
        )
        self.fc = nn.Linear(self.n_features, self.n_actions)

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract_features(x)
        x = self.fc(x)
        return x

    def act(
        self, x: Tensor, action_masks: Tensor, deterministic: bool = False
    ) -> Tensor:
        action_logits = self.forward(x)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()


class DictFeatureNet(nn.Module):
    def __init__(self, n_conv_layers: int, n_pass_through_layers: int, n_features: int, n_actions: int):
        super().__init__()
        LAYER_WIDTH = 12
        LAYER_HEIGHT = 12

        self.n_actions = n_actions
        self.n_features = 128
        n_observables = 13
        self.net = nn.Sequential(
            nn.Linear(n_observables, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_features),
            nn.Tanh(),
        )
        self.fc = nn.Linear(self.n_features, self.n_actions)

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract_features(x)
        x = self.fc(x[:, :self.n_features])
        return x

    def act(
        self, x: Tensor, action_masks: Tensor, deterministic: bool = False
    ) -> Tensor:
        action_logits = self.forward(x)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.mode if deterministic else dist.sample()

from gym import spaces
from luxai_s2.state.state import ObservationStateDict
import numpy as np
from torch import nn
from torch.functional import Tensor
from typing import Dict, Any

from lux_entry.lux.state import Player


class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        "Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server"
        self.action_space = action_space

    def action_to_lux_action(self, player: Player, obs: ObservationStateDict, action: np.ndarray) -> Dict[str, int]:
        "Takes the observation and the parameterized action and returns an action formatted for the Lux env"
        raise NotImplementedError()

    def action_masks(self, player: Player, obs: ObservationStateDict) -> np.ndarray:
        "Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not"
        raise NotImplementedError()


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def act(
        self, x: Tensor, action_masks: Tensor, deterministic: bool = False
    ) -> Tensor:
        raise NotImplementedError()

    def load_weights(self, state_dict: Any) -> None:
        raise NotImplementedError()

from gym import spaces
from luxai_s2.state.state import ObservationStateDict
import numpy as np
from typing import Dict


class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        "Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server"
        self.action_space = action_space

    def action_to_lux_action(self, agent: str, obs: Dict[str, ObservationStateDict], action: np.ndarray) -> Dict[str, int]:
        "Takes the observation and the parameterized action and returns an action formatted for the Lux env"
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, ObservationStateDict]) -> np.ndarray:
        "Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not"
        raise NotImplementedError()

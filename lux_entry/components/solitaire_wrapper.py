import gym
import numpy as np
from typing import Dict, Tuple, Any

from luxai_s2.state import ObservationStateDict

from lux_entry.lux.state import Player


class SolitaireWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        """
        This wrapper makes the step() function take a single action for a single player, and return single-player values.
        Opponents don't get an action.
        """
        super().__init__(env)
        self.env = env
        self.player = player

    def step(self, action: np.ndarray) -> Tuple[
        ObservationStateDict,  # obs
        float,  # reward
        bool,  # done
        Any,  # info
    ]:
        dual_action = {self.player: action}  # the second player doesn't get an action
        obs, reward, done, info = self.env.step(dual_action)
        return obs[self.player], reward[self.player], done[self.player], info[self.player]

    def reset(self, **kwargs) -> Dict[Player, ObservationStateDict]:
        obs = self.env.reset(**kwargs)
        return obs[self.player]

import gym
import numpy as np
from typing import Callable, Dict, Tuple, Any

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from lux_entry.heuristics.bidding import BidActionType
from lux_entry.heuristics.factory_placement import FactoryPlacementActionType
from lux_entry.lux.state import Player
from lux_entry.lux.utils import my_turn_to_place_factory
from lux_entry.components.types import Controller


class BaseWrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[[Player, ObservationStateDict], BidActionType],
        factory_placement_policy: Callable[
            [Player, ObservationStateDict], FactoryPlacementActionType
        ],
        controller: Controller,
    ) -> None:
        """
        This wrapper goes around LuxAI_S2, which is directly called whenever self.env is invoked.
        LuxAI_S2 takes actions and outputs transitions for both agents simultaneously, in the form Dict[Player, Any].
        This wrapper takes a bid and factory placement policy, which both players use to play the first two game phases on reset.
        The wrapper also takes an action controller, which is used to set the action space and convert to LuxAI_S2 actions on step.
        """
        super().__init__(env)
        self.env = env
        self.controller = controller
        self.action_space = controller.action_space
        self.factory_placement_policy = factory_placement_policy
        self.bid_policy = bid_policy
        self.prev_obs = None

    def step(
        self, player_actions: Dict[Player, np.ndarray]
    ) -> Tuple[
        Dict[Player, ObservationStateDict],  # obs
        Dict[Player, float],  # reward
        Dict[Player, bool],  # done
        Dict[Player, Any],  # info
    ]:
        """
        Actions for one or more players are passed in, and are converted to Lux actions.
        If the input only contains an action for one player, the other gets an empty dict.
        The actions is fed to LuxAI_S2, which returns a transition for both players.
        """
        # here, for each player in the game we translate their action into a Lux S2 action
        lux_action = dict()
        for player in self.env.agents:
            if player in player_actions:
                assert self.prev_obs is not None
                lux_action[player] = self.controller.action_to_lux_action(
                    player=player, obs=self.prev_obs[player], action=player_actions[player]
                )
            else:
                lux_action[player] = dict()

        # lux_action is now a dict mapping player name to an action, which is passed to LuxAI_S2
        obs, reward, done, info = self.env.step(lux_action)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs) -> Dict[Player, ObservationStateDict]:
        """
        Reset the LuxAI_S2 environment first.
        Then both players use the provided bid and factory placement policies to play the first two game phases.
        """
        # we call the original reset function first
        obs = self.env.reset(**kwargs)

        # then use the bid policy to go through the bidding phase
        player_actions = dict()
        for agent in self.env.agents:
            player_actions[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(player_actions)

        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
        while self.env.state.real_env_steps < 0:
            player_actions = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    player_actions[agent] = self.factory_placement_policy(
                        agent, obs[agent]
                    )
                else:
                    player_actions[agent] = dict()
            obs, _, _, _ = self.env.step(player_actions)
        self.prev_obs = obs

        return obs

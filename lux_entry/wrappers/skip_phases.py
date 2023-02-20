import copy
import gym
import numpy.typing as npt
from typing import Callable, Dict

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from lux_entry.heuristics.bidding import BidActionType
from lux_entry.heuristics.factory_placement import FactoryPlacementActionType
from lux_entry.lux.state import Player
from lux_entry.lux.stats import StatsStateDict
from lux_entry.lux.utils import my_turn_to_place_factory
from lux_entry.wrappers.controllers import Controller


class MainGameOnlyWrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[[Player, ObservationStateDict], BidActionType],
        factory_placement_policy: Callable[[Player, ObservationStateDict], FactoryPlacementActionType],
        controller: Controller,
    ) -> None:
        """
        Sets the bidding and factory placement policies.
        """
        super().__init__(env)
        self.env = env
        self.controller = controller
        self.action_space = controller.action_space
        self.factory_placement_policy = factory_placement_policy
        self.bid_policy = bid_policy
        self.prev_obs = None

    def step(self, action: Dict[str, npt.NDArray]):
        # here, for each agent in the game we translate their action into a Lux S2 action
        lux_action = dict()
        for agent in self.env.agents:
            if agent in action:
                assert self.prev_obs is not None
                lux_action[agent] = self.controller.action_to_lux_action(
                    agent=agent, obs=self.prev_obs, action=action[agent]
                )
            else:
                lux_action[agent] = dict()

        # lux_action is now a dict mapping agent name to an action
        obs, reward, done, info = self.env.step(lux_action)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        # we call the original reset function first
        obs = self.env.reset(**kwargs)

        # then use the bid policy to go through the bidding phase
        action = dict()
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)

        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    # TODO: make sure players aren't placing when it's not their turn
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent] = self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, _, _, _ = self.env.step(action)
        self.prev_obs = obs

        return obs


class MainGameOnlyEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training"""
        super().__init__(env)
        self.prev_step_metrics = None
        self.player = "player_0"
        self.opp_player = "player_1"

    def step(self, action):
        # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
        for factory in self.env.state.factories[self.opp_player].values():
            factory.cargo.water = 1000

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {self.player: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[self.player]
        done = done[self.player]

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[self.player]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs

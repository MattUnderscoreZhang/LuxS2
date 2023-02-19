import copy
import gym
from gym.wrappers.time_limit import TimeLimit
from typing import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from luxai_s2 import LuxAI_S2
from luxai_s2.state import StatsStateDict

from lux_entry.heuristics import bidding, factory_placement
from lux_entry.wrappers.controllers import ControllerWrapper
from lux_entry.wrappers.observations import ObservationWrapper
from lux_entry.wrappers.skip_phases import MainGameOnlyWrapper


class EnvWrapper(gym.Wrapper):
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


def make_env(
    env_id: str, rank: int, seed: int = 0, max_episode_steps: int = 100
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
        env = MainGameOnlyWrapper(
            env,
            bid_policy=bidding.zero_bid,
            factory_placement_policy=factory_placement.place_near_random_ice,
            controller=ControllerWrapper(env.env_cfg),
        )
        env = ObservationWrapper(
            env
        )  # changes observation to include a few simple features
        env = EnvWrapper(env)  # convert to single agent, add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init

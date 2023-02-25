import gym
from gym.wrappers.time_limit import TimeLimit
from typing import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from lux_entry.heuristics import bidding, factory_placement
from lux_entry.wrappers import controllers
from lux_entry.wrappers import observations
from lux_entry.wrappers.game import MainGameOnlyWrapper, SinglePlayerWrapper


bid_policy = bidding.zero_bid
factory_placement_policy = factory_placement.place_near_random_ice
controller = controllers.single_unit_controller
observation_wrapper = observations.starter_kit_observations


def make_env(
    env_id: str, rank: int, seed: int = 0, max_episode_steps: int = 100
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
        env = MainGameOnlyWrapper(
            env,
            bid_policy=bid_policy,
            factory_placement_policy=factory_placement_policy,
            controller=controller.Controller(env.env_cfg),
        )
        env = observation_wrapper.ObservationWrapper(env)
        env = SinglePlayerWrapper(env)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init

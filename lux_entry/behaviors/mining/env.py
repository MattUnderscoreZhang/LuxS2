import copy
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from typing import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from luxai_s2.state.state import ObservationStateDict

from lux_entry.components.base_wrapper import BaseWrapper
from lux_entry.components.map_features_obs import get_full_obs_space
from lux_entry.components.solitaire_wrapper import SolitaireWrapper
from lux_entry.components.types import Controller, PolicyNet
from lux_entry.heuristics import bidding, factory_placement
from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player
from lux_entry.lux.stats import StatsStateDict
from lux_entry.lux.utils import add_batch_dimension

from . import controller, observations


class EnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training.
        Only a single player's action is passed to step. Only single-player transitions are returned.
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.player = "player_0"
        self.opp_player = "player_1"

    def step(self, action: np.ndarray):
        """
        Update the environment state directly to keep the enemy alive.
        Set enemy factories to have 1000 water so the game can be treated as single-agent.
        Send a single-player action and return a single-player transition.
        We calculate our own reward and info, where info["metrics"] is used with Tensorboard in train.py.
        """
        # keep the enemy alive
        for factory in self.env.state.factories[self.opp_player].values():
            factory.cargo.water = 1000

        # step
        obs, _, done, info = self.env.step(action)

        # calculate metrics
        stats: StatsStateDict = self.env.state.stats[self.player]
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]
        info["metrics"] = metrics

        # calculate reward
        reward = 0
        if self.prev_step_metrics is not None:
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            reward = ice_dug_this_step / 100 + water_produced_this_step  # water is more important
        self.prev_step_metrics = copy.deepcopy(metrics)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        After reset we only return a single-player observation.
        """
        obs = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return obs


bid_policy = bidding.zero_bid
factory_placement_policy = factory_placement.place_near_random_ice
ObservationWrapper = observations.ObservationWrapper
EnvController = controller.EnvController


def make_env(
    rank: int, seed: int = 0, max_episode_steps: int = 100
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        """
        This environment is only used during training.
        We overwrite the reset and step functions via wrappers.
        The observation and action functions can also be overwritten via wrappers.
        """
        env = gym.make(id="LuxAI_S2-v0", verbose=0, collect_stats=True, MAX_FACTORIES=2)
        env = BaseWrapper(
            env,
            bid_policy=bid_policy,
            factory_placement_policy=factory_placement_policy,
            controller=EnvController(env.env_cfg),
        )
        env = SolitaireWrapper(env, "player_0")
        env = ObservationWrapper(env)
        env = EnvWrapper(env)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


def act(
    step: int,
    env_obs: ObservationStateDict,
    remainingOverageTime: int,
    player: Player,
    env_cfg: EnvConfig,
    controller: Controller,
    net: PolicyNet
):
    observation_space = get_full_obs_space(env_cfg)
    obs = observations.ObservationWrapper.get_obs(env_obs, env_cfg, observation_space)

    with torch.no_grad():
        action_mask = add_batch_dimension(
            controller.action_masks(player=player, obs=env_obs)
        ).bool()
        observation = add_batch_dimension(obs)
        actions = (
            net.act(observation, deterministic=False, action_masks=action_mask)
            .cpu()
            .numpy()
        )
    return controller.action_to_lux_action(player, env_obs, actions[0])

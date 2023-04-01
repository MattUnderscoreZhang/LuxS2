import copy
import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from typing import Callable, Any, Union

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from lux_entry.heuristics import bidding, factory_placement
from lux_entry.lux.state import Player
from lux_entry.lux.stats import StatsStateDict
from lux_entry.lux.utils import my_turn_to_place_factory
from lux_entry.training.controller import EnvController
from lux_entry.training.observations import (
    get_full_obs,
    get_minimap_obs,
    get_obs_by_job,
)


bid_policy = bidding.zero_bid
factory_placement_policy = factory_placement.place_near_random_ice
controller = EnvController


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
            controller=controller(env.env_cfg),
        )
        env = SolitaireWrapper(env, "player_0")
        env = ObservationWrapper(env, "player_0")
        env = TrainingWrapper(env, "player_0")
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


class BaseWrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[[Player, ObservationStateDict], bidding.BidActionType],
        factory_placement_policy: Callable[
            [Player, ObservationStateDict], factory_placement.FactoryPlacementActionType
        ],
        controller: EnvController,
    ) -> None:
        """
        Uses bid_policy and factory_placement_policy to auto-play the first two game phases.
        Uses controller to convert actions to LuxAI_S2 actions.
        Takes actions and outputs transitions for both players simultaneously.
        """
        super().__init__(env)
        self.env = env
        self.controller = controller
        self.action_space = controller.action_space
        self.factory_placement_policy = factory_placement_policy
        self.bid_policy = bid_policy
        self.prev_obs = None

    def step(
        self, player_actions: dict[Player, np.ndarray]
    ) -> tuple[
        dict[Player, ObservationStateDict],  # obs
        dict[Player, float],  # reward
        dict[Player, bool],  # done
        dict[Player, Any],  # info
    ]:
        """
        Actions for one or more players are passed in, and are converted to Lux actions.
        If the input only contains an action for one player, the other gets an empty dict.
        The actions are fed to LuxAI_S2, which returns a transition for both players.
        """
        # here, for each player in the game we translate their action into a Lux S2 action
        lux_actions = dict()
        for player in self.env.agents:
            if player in player_actions:
                assert self.prev_obs is not None
                lux_actions[player] = self.controller.action_to_lux_action(
                    player=player, obs=self.prev_obs[player], action=player_actions[player]
                )
            else:
                lux_actions[player] = dict()

        # lux_actions is now a dict mapping player name to an action, which is passed to LuxAI_S2
        obs, reward, done, info = self.env.step(lux_actions)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs) -> dict[Player, ObservationStateDict]:
        """
        Reset the LuxAI_S2 environment.
        The bid and factory placement policies are used to play the first two game phases.
        Both players use the same policies.
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
                    obs["player_0"]["teams"][agent]["place_first"],  # TODO: make sure this is alternating players
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


class SolitaireWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        """
        step() takes a single player action and returns a single-player transition.
        An empty action is filled in for the opponent.
        """
        super().__init__(env)
        self.env = env
        self.player = player

    def step(self, action: np.ndarray) -> tuple[
        ObservationStateDict,  # obs
        float,  # reward
        bool,  # done
        Any,  # info
    ]:
        dual_action = {self.player: action}  # the second player doesn't get an action
        obs, reward, done, info = self.env.step(dual_action)
        return obs[self.player], reward[self.player], done[self.player], info[self.player]

    def reset(self, **kwargs) -> dict[Player, ObservationStateDict]:
        obs = self.env.reset(**kwargs)
        return obs[self.player]


UnitObsInfo = dict[str, Union[str, torch.Tensor]]


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        self.observation_space = spaces.Dict({
            "conv_obs":spaces.Box(-999, 999, shape=(104, 12, 12)),
            "skip_obs":spaces.Box(-999, 999, shape=(4, 12, 12)),
        })
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"

    def observation(
        self, obs: ObservationStateDict
    ) -> dict[str, UnitObsInfo]:
        """
        Get minimaps for each unit based on what it needs to know for its job.
        """
        full_obs = get_full_obs(obs, self.env_cfg, self.player, self.opponent)
        assert full_obs.has_ice.shape == (1, 48, 48)

        units = obs["units"][self.player]
        unit_jobs = {
            unit_info["unit_id"]: "general"  # TODO: calculate unit jobs
            for unit_info in units.values()
        }
        minimap_obs = {}
        for unit_info in units.values():
            unit_id = unit_info["unit_id"]
            full_obs_subset = get_obs_by_job(full_obs, unit_jobs[unit_id])
            mini_obs = get_minimap_obs(full_obs_subset, unit_info["pos"])
            minimap_obs[unit_id] = {
                "job": unit_jobs[unit_id],
                "mini_obs": mini_obs,
            }
        return minimap_obs


class TrainingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        """
        Alters the environment between steps for training purposes.
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"

    def step(self, action: np.ndarray):
        """
        Update the environment state directly to keep the enemy alive.
        Set enemy factories to have 1000 water so the game can be treated as single-agent.
        Send a single-player action and return a single-player transition.
        Calculate reward and info, with info["metrics"] passed to Tensorboard in train.py.
        """
        # keep the enemy alive
        for factory in self.env.state.factories[self.opponent].values():
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
        reward = 0  # TODO: pass in reward calculation function during TrainingWrapper init
        if self.prev_step_metrics is not None:
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            reward = ice_dug_this_step / 100 + water_produced_this_step  # prioritize water
        self.prev_step_metrics = copy.deepcopy(metrics)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        After reset we only return a single-player observation.
        """
        obs = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return obs

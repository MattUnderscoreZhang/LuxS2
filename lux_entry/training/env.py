import copy
import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from typing import Callable, Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from lux_entry.heuristics import bidding, factory_placement
from lux_entry.lux.state import Player
from lux_entry.lux.stats import StatsStateDict
from lux_entry.lux.utils import my_turn_to_place_factory
from lux_entry.training.controller import EnvController
from lux_entry.training.observations import get_full_obs, get_minimap_obs


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
        env = TrainingWrapper(env)
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
        This wrapper goes around LuxAI_S2, which is directly called whenever self.env is invoked.
        LuxAI_S2 takes actions and outputs transitions for both agents simultaneously, in the form dict[Player, Any].
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

    def reset(self, **kwargs) -> dict[Player, ObservationStateDict]:
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


class SolitaireWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        """
        This wrapper makes the step() function take a single action for a single player, and return single-player values.
        Opponents don't get an action.
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


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        self.observation_space = spaces.Dict({
            "conv_obs":spaces.Box(-999, 999, shape=(104, 12, 12)),
            "skip_obs":spaces.Box(-999, 999, shape=(4, 12, 12)),
        })
        self.player = player

    def observation(self, obs: ObservationStateDict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get minimaps.
        """
        full_obs = get_full_obs(obs, self.env_cfg)
        assert full_obs.tile_has_ice.shape == (1, 48, 48)
        obs_to_process = [
            (full_obs.tile_has_ice, True),
            (full_obs.tile_per_player_has_factory, False),
            (full_obs.tile_per_player_has_robot, False),
            (full_obs.tile_per_player_has_light_robot, False),
            (full_obs.tile_per_player_has_heavy_robot, False),
            (full_obs.tile_rubble, False),
            (full_obs.tile_per_player_light_robot_power, False),
            (full_obs.tile_per_player_heavy_robot_power, False),
            (full_obs.tile_per_player_factory_ice_unbounded, False),
            (full_obs.tile_per_player_factory_ore_unbounded, False),
            (full_obs.tile_per_player_factory_water_unbounded, False),
            (full_obs.tile_per_player_factory_metal_unbounded, False),
            (full_obs.tile_per_player_factory_power_unbounded, False),
            (full_obs.game_is_day, False),
            (full_obs.game_day_or_night_elapsed, False),
        ]

        units = obs["units"][self.player]
        minimap_obs = {
            unit_info["unit_id"]: get_minimap_obs(obs_to_process, unit_info["pos"])
            for unit_info in units.values()
        }
        return minimap_obs


class TrainingWrapper(gym.Wrapper):
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

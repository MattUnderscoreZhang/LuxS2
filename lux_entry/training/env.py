import gym
from gym.wrappers.time_limit import TimeLimit
import numpy as np
from torch import Tensor
from typing import Callable, Any, Dict, Tuple

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from lux_entry.heuristics import bidding, factory_placement
from lux_entry.lux.state import Player
from lux_entry.lux.stats import StatsStateDict
from lux_entry.lux.utils import my_turn_to_place_factory
from lux_entry.training.controller import EnvController
from lux_entry.training.observations import get_full_obs, get_full_obs_space
from lux_entry.training.rewards import ice_mining_reward


bid_policy = bidding.zero_bid
factory_placement_policy = factory_placement.random_factory_placement
controller = EnvController


def make_env(
    rank: int, seed: int = 0, max_episode_steps: int = 100
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        """
        This environment is only used during training.
        We overwrite reset, step, observation, and action functions via wrappers.
        """
        env = gym.make(id="LuxAI_S2-v0", verbose=0, collect_stats=True, MAX_FACTORIES=2)
        env = BaseWrapper(
            env,
            bid_policy=bid_policy,
            factory_placement_policy=factory_placement_policy,
            controller=controller(env.env_cfg),
        )
        env = SolitaireWrapper(env, "player_0")
        env = RewardWrapper(env, "player_0", ice_mining_reward)
        env = FullObservationWrapper(env, "player_0")
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
        Takes two-player actions and outputs two-player transitions.
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
        The actions are fed to LuxAI_S2, which returns a transition for both players.
        """
        lux_actions = {
            player: (
                self.controller.actions_to_lux_actions(
                    player=player, obs=self.prev_obs[player], actions=player_actions[player]
                )
                if player in player_actions.keys() and self.prev_obs is not None
                else dict()
            )
            for player in self.env.agents
        }
        # lux_actions passed to LuxAI_S2 env
        obs, reward, done, info = self.env.step(lux_actions)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs) -> Dict[Player, ObservationStateDict]:
        """
        Reset the LuxAI_S2 environment.
        The bid and factory placement policies are used to play the first two game phases.
        Both players use the same policies.
        """
        # we call the original reset function first
        obs = self.env.reset(**kwargs)

        # then use the bid policy to go through the bidding phase
        player_actions = {
            player: self.bid_policy(player, obs[player])
            for player in self.env.agents
        }
        obs, _, _, _ = self.env.step(player_actions)

        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
        while self.env.state.real_env_steps < 0:
            player_actions = {
                player: (
                    self.factory_placement_policy(player, obs[player])
                    if my_turn_to_place_factory(
                        obs[player]["teams"][player]["place_first"],
                        self.env.state.env_steps,
                    )
                    else dict()
                )
                for player in self.env.agents
            }
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

    def step(self, action: np.ndarray) -> Tuple[
        ObservationStateDict,  # obs
        float,  # reward
        bool,  # done
        Any,  # info
    ]:
        player_actions = {self.player: action}  # the second player doesn't get an action
        obs, reward, done, info = self.env.step(player_actions)
        return obs[self.player], reward[self.player], done[self.player], info[self.player]

    def reset(self, **kwargs) -> Dict[Player, ObservationStateDict]:
        obs = self.env.reset(**kwargs)
        return obs[self.player]


class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, player: Player, reward: Callable) -> None:
        """
        Alters the environment between steps for training purposes.
        """
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        self.prev_reward_calculations = None
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"
        self.reward = reward

    def keep_enemy_alive(self):
        """
        Update the environment state directly to keep the enemy alive.
        Set enemy factories to have 1000 water so the game can be treated as single-agent.
        """
        # keep the enemy alive
        for factory in self.env.state.factories[self.opponent].values():
            factory.cargo.water = 1000

    def calculate_metrics(self) -> Dict:
        stats: StatsStateDict = self.env.state.stats[self.player]
        metrics = {
            "total_ice_dug": (
                stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
            ),
            "total_water_produced": stats["generation"]["water"],
            "action_queue_updates_success": stats["action_queue_updates_success"],
            "action_queue_updates_total": stats["action_queue_updates_total"],
        }
        return metrics

    def step(self, action: np.ndarray):
        """
        Calculate metrics and reward, with info["metrics"] passed to Tensorboard in train.py.
        """
        self.keep_enemy_alive()
        obs, _, done, info = self.env.step(action)
        info["metrics"] = self.calculate_metrics()
        reward, self.prev_reward_calculations = self.reward(
            obs=obs,
            player=self.player,
            env_cfg=self.env_cfg,
            prev_reward_calculations=self.prev_reward_calculations,
        )
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return obs


class FullObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, player: Player) -> None:
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        self.observation_space = get_full_obs_space(self.env_cfg)
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"

    def observation(self, obs: ObservationStateDict) -> Dict[str, Tensor]:
        return get_full_obs(obs, self.env_cfg, self.player, self.opponent)

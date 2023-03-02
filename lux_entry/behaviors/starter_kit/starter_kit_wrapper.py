import copy
import gym
import numpy as np

from lux_entry.lux.stats import StatsStateDict


class StarterKitWrapper(gym.Wrapper):
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
        dual_action = {
            self.player: action
        }  # the second player doesn't get an action
        obs, _, done, info = self.env.step(dual_action)
        obs = obs[self.player]
        done = done[self.player]

        # calculate metrics
        stats: StatsStateDict = self.env.state.stats[self.player]
        info = dict()
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
        return obs[self.player]

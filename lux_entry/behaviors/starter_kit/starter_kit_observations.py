import gym
from gym import spaces
import numpy as np
from typing import Dict

from luxai_s2.state.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        This wrapper returns a dual-player observation with some simple calculated features.
        """
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(self, two_player_env_obs: Dict[Player, ObservationStateDict]):
        """
        This takes in a dual-player observation from the underlying BaseWrapper environment.
        The custom obs calculation function is static so the submission/evaluation code can use it.
        The custom obs is returned to the wrapper around this one when it calls step.
        """
        return ObservationWrapper.get_custom_obs(two_player_env_obs, self.env.state.env_cfg)

    @staticmethod
    def get_custom_obs(
        two_player_env_obs: Dict[Player, ObservationStateDict], env_cfg: EnvConfig
    ) -> Dict[Player, np.ndarray]:
        """
        Return obs contains info on your first robot, your first factory, and some useful features.
        No information about the opponent is included. This returns a set of obs for each player.
        If there are no owned robots the observation is just zero.

        Included features:
        - First robot's stats
        - distance vector to closest ice tile
        - distance vector to first factory
        """
        observation = dict()
        env_obs = two_player_env_obs["player_0"]
        ice_map = env_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in two_player_env_obs.keys():
            factory_vec = np.zeros(2)
            for factory in env_obs["factories"][agent].values():
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break  # just the first factory

            obs_vec = np.zeros(13,)
            for unit in env_obs["units"][agent].values():
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array([
                    unit["power"] / battery_cap,
                    unit["cargo"]["ice"] / cargo_space,
                    unit["cargo"]["ore"] / cargo_space,
                    unit["cargo"]["water"] / cargo_space,
                    unit["cargo"]["metal"] / cargo_space,
                ])
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # add distance to closest ice tile and to factory
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                break  # just the first unit

            observation[agent] = obs_vec

        return observation

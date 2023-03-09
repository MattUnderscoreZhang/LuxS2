import gym
from gym import spaces
import numpy as np
import torch
from typing import Dict

from luxai_s2.state.state import ObservationStateDict

from lux_entry.components.map_features_obs import MapFeaturesObservation, get_full_obs_space, get_full_obs
from lux_entry.lux.state import EnvConfig


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        self.observation_space = spaces.Dict({
            "conv_obs":spaces.Box(-999, 999, shape=(104, 12, 12)),
            "skip_obs":spaces.Box(-999, 999, shape=(4, 12, 12)),
        })

    def observation(self, obs: ObservationStateDict) -> Dict[str, torch.Tensor]:
        return ObservationWrapper.get_obs(obs, self.env_cfg, get_full_obs_space(self.env_cfg))

    @staticmethod
    def _concat_obs(conv_obs: list[np.ndarray], skip_obs: list[np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Concat observables together into arrays.
        """
        for obs in conv_obs + skip_obs:
            assert (
                len(obs.shape) == 3
                # variable second dimension
                and obs.shape[1] == 12
                and obs.shape[2] == 12
            )
        return {
            "conv_obs": torch.cat([torch.from_numpy(obs) for obs in conv_obs], dim=0),
            "skip_obs": torch.cat([torch.from_numpy(obs) for obs in skip_obs], dim=0),
        }

    @staticmethod
    def _mean_pool(arr: np.ndarray, window: int) -> np.ndarray:
        arr = arr.reshape(
            arr.shape[0] // window, window, arr.shape[1] // window, window
        )
        return np.mean(arr, axis=(1, 3))

    @staticmethod
    def _get_minimaps(full_obs: MapFeaturesObservation, x: int, y: int) -> Dict[str, torch.Tensor]:
        """
        Create minimaps for a set of features around (x, y).
        """
        # observables to get minimaps for, as (observable, skip_obs)
        minimap_obs = [
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

        # create minimaps centered around x, y
        conv_minimaps = []
        skip_minimaps = []
        for value, skip in minimap_obs:
            expanded_map = np.full((value.shape[0], 96, 96), -1.0)
            minimap = np.zeros((value.shape[0] * 4, 12, 12))
            for p in range(value.shape[0]):
                # unit is in lower right pixel of upper left quadrant
                expanded_map[p][x : x + 48, y : y + 48] = value[p]
                # small map (12x12 area)
                minimap[p * 4] = expanded_map[p][42:54, 42:54]
                # medium map (24x24 area)
                minimap[p * 4 + 1] = ObservationWrapper._mean_pool(expanded_map[p][36:60, 36:60], 2)
                # large map (48x48 area)
                minimap[p * 4 + 2] = ObservationWrapper._mean_pool(expanded_map[p][24:72, 24:72], 4)
                # full map (96x96 area)
                minimap[p * 4 + 3] = ObservationWrapper._mean_pool(expanded_map[p], 8)
            conv_minimaps.append(minimap)
            if skip:
                skip_minimaps.append(minimap)

        return ObservationWrapper._concat_obs(conv_minimaps, skip_minimaps)

    @staticmethod
    def get_obs(
        obs: ObservationStateDict,
        env_cfg: EnvConfig,
        observation_space: spaces.Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Get minimaps.
        """
        full_obs = get_full_obs(obs, env_cfg, observation_space)
        assert full_obs.tile_has_ice.shape == (1, 48, 48)

        first_unit_obs = {
            "conv_obs": torch.zeros((104, 12, 12)),
            "skip_obs": torch.zeros((4, 12, 12)),
        }
        units = obs["units"]
        for player in ["player_0", "player_1"]:
            units = obs["units"][player]
            for unit_info in units.values():
                first_unit_obs = ObservationWrapper._get_minimaps(
                    full_obs, unit_info["pos"][0], unit_info["pos"][1]
                )
                break  # get just first unit

        return first_unit_obs

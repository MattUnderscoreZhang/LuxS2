from gym import spaces
import numpy as np
import torch
from torch import Tensor

from luxai_s2.state.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player


N_OBS_CHANNELS = 56


per_map_tile_obs_keys = [
    # binary yes/no
    "has_ice",
    "has_ore",
    "player_has_factory",
    "player_has_robot",
    "player_has_light_robot",
    "player_has_heavy_robot",
    "player_has_lichen",
    # normalized from 0-1
    "rubble",
    "player_lichen",
    # normalized from 0-1, -1 means inapplicable
    "player_robot_power",
    "player_robot_ice",
    "player_robot_ore",
    "player_robot_cargo",
    # normalized and positive unbound, -1 means inapplicable
    "player_factory_ice_unb",
    "player_factory_ore_unb",
    "player_factory_water_unb",
    "player_factory_metal_unb",
    "player_factory_power_unb",
]
broadcast_obs_keys = [
    # normalized and positive unbound
    "player_tot_robots_unb",
    "player_tot_light_robots_unb",
    "player_tot_heavy_robots_unb",
    "player_tot_factory_ice_unb",
    "player_tot_factory_ore_unb",
    "player_tot_factory_water_unb",
    "player_tot_factory_metal_unb",
    "player_tot_factory_power_unb",
    "player_tot_lichen_unb",
    # normalized from 0-1
    "player_tot_factories",
    "game_is_day",
    "game_day_or_night_elapsed",
    "game_time_elapsed",
]
obs_keys = per_map_tile_obs_keys + broadcast_obs_keys


def get_full_obs_space(env_cfg: EnvConfig) -> spaces.Dict:
    map_size = env_cfg.map_size
    spaces_dict = spaces.Dict()
    for key in obs_keys:
        n_features = 2 if "player_" in key else 1
        low = (
            -1.0  # not applicable (no robot or factory on tile)
            if (("_robot_" in key or "_factory_" in key) and "_tot_" not in key)
            else 0.0
        )
        high = np.inf if "_unb" in key else 1.0
        spaces_dict[key] = (
            spaces.MultiBinary((n_features, map_size, map_size))
            if "has_" in key or "is_" in key
            else spaces.Box(low, high, shape=(n_features, map_size, map_size))
        )
    return spaces_dict


def get_full_obs(
    env_obs: ObservationStateDict, env_cfg: EnvConfig, player: Player, opponent: Player,
) -> dict[str, Tensor]:
    # normalization factors
    MAX_FACS = env_cfg.MAX_FACTORIES
    MAX_RUBBLE = env_cfg.MAX_RUBBLE
    MAX_LICHEN = env_cfg.MAX_LICHEN_PER_TILE
    L_CAR_SPACE = env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
    L_BAT_CAP = env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY
    H_CAR_SPACE = env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
    H_BAT_CAP = env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY
    EXP_FAC_CAR = 2 * H_CAR_SPACE
    ICE_WATER_RATIO = env_cfg.ICE_WATER_RATIO
    ORE_METAL_RATIO = env_cfg.ORE_METAL_RATIO
    EXP_FAC_WATER = EXP_FAC_CAR / ICE_WATER_RATIO
    EXP_FAC_METAL = EXP_FAC_CAR / ORE_METAL_RATIO
    EXP_TOT_FAC_CAR = MAX_FACS * EXP_FAC_CAR
    EXP_TOT_FAC_WATER = EXP_TOT_FAC_CAR / ICE_WATER_RATIO
    EXP_TOT_FAC_METAL = EXP_TOT_FAC_CAR / ORE_METAL_RATIO
    EXP_FAC_POWER = env_cfg.FACTORY_CHARGE * 30
    EXP_TOT_FAC_POWER = EXP_FAC_POWER * MAX_FACS
    EXP_RS = 50
    CYCLE_LEN = env_cfg.CYCLE_LENGTH
    DAY_LEN = env_cfg.DAY_LENGTH
    MAX_EPISODE_LEN = env_cfg.max_episode_length

    # init keys in observation as array of zeros of the right size
    obs = {
        key: np.zeros(value.shape)
        if type(value) == spaces.MultiBinary
        else np.zeros(value.shape) + value.low[0]
        for key, value in get_full_obs_space(env_cfg).items()
    }

    # fill in observations
    obs["has_ice"][0] = env_obs["board"]["ice"]
    obs["has_ore"][0] = env_obs["board"]["ore"]
    obs["rubble"][0] = env_obs["board"]["rubble"] / MAX_RUBBLE
    lichen_strains = [[], []]
    for p, player in enumerate([player, opponent]):
        for fac in env_obs["factories"][player].values():
            cargo = fac["cargo"]
            p_pos = (p, fac["pos"][0], fac["pos"][1])
            lichen_strains[p].append(fac["strain_id"])
            obs["player_has_factory"][p_pos] = 1
            obs["player_factory_ice_unb"][p_pos] = cargo["ice"] / EXP_FAC_CAR
            obs["player_factory_ore_unb"][p_pos] = cargo["ore"] / EXP_FAC_CAR
            obs["player_factory_water_unb"][p_pos] = cargo["water"] / EXP_FAC_WATER
            obs["player_factory_metal_unb"][p_pos] = cargo["metal"] / EXP_FAC_METAL
            obs["player_factory_power_unb"][p_pos] = fac["power"] / EXP_FAC_POWER
            obs["player_tot_factory_ice_unb"][p] += cargo["ice"] / EXP_TOT_FAC_CAR
            obs["player_tot_factory_ore_unb"][p] += cargo["ore"] / EXP_TOT_FAC_CAR
            obs["player_tot_factory_water_unb"][p] += cargo["water"] / EXP_TOT_FAC_WATER
            obs["player_tot_factory_metal_unb"][p] += cargo["metal"] / EXP_TOT_FAC_METAL
            obs["player_tot_factory_power_unb"][p] += fac["power"] / EXP_TOT_FAC_POWER
        for unit in env_obs["units"][player].values():
            cargo = unit["cargo"]
            p_pos = (p, unit["pos"][0], unit["pos"][1])
            obs["player_has_robot"][p_pos] = 1
            if unit["unit_type"] == "LIGHT":
                obs["player_has_light_robot"][p_pos] = 1
                obs["player_robot_power"][p_pos] = unit["power"] / L_BAT_CAP
                obs["player_robot_ice"][p_pos] = cargo["ice"] / L_CAR_SPACE
                obs["player_robot_ore"][p_pos] = cargo["ore"] / L_CAR_SPACE
                obs["player_robot_cargo"][p_pos] = (cargo["ice"] + cargo["ore"]) / L_CAR_SPACE
                obs["player_tot_light_robots_unb"][p] += 1 / EXP_RS
            elif unit["unit_type"] == "HEAVY":
                obs["player_has_heavy_robot"][p_pos] = 1
                obs["player_robot_power"][p_pos] = unit["power"] / H_BAT_CAP
                obs["player_robot_ice"][p_pos] = cargo["ice"] / H_CAR_SPACE
                obs["player_robot_ore"][p_pos] = cargo["ore"] / H_CAR_SPACE
                obs["player_robot_cargo"][p_pos] = (cargo["ice"] + cargo["ore"]) / H_CAR_SPACE
                obs["player_tot_heavy_robots_unb"][p] += 1 / EXP_RS
        obs["player_tot_factories"][p] += len(env_obs["factories"][player]) / MAX_FACS
        obs["player_tot_robots_unb"][p] += len(env_obs["units"][player]) / EXP_RS
        obs["player_has_lichen"][p] = env_obs["board"]["lichen_strains"] == lichen_strains[p]
        obs["player_lichen"][p] = (
            env_obs["board"]["lichen"] / MAX_LICHEN
            * (env_obs["board"]["lichen_strains"] == lichen_strains[p])
        )
        obs["player_tot_lichen_unb"][p] += np.sum(obs["player_lichen"][p])
    game_is_day = env_obs["real_env_steps"] % CYCLE_LEN < DAY_LEN
    obs["game_is_day"][0] += game_is_day
    obs["game_day_or_night_elapsed"][0] += (
        (env_obs["real_env_steps"] % CYCLE_LEN) / DAY_LEN
        if game_is_day
        else (env_obs["real_env_steps"] % CYCLE_LEN - DAY_LEN)
        / (CYCLE_LEN - DAY_LEN)
    )
    obs["game_time_elapsed"][0] += env_obs["real_env_steps"] / MAX_EPISODE_LEN
    # assert [k for k in obs.keys()] == obs_keys
    # assert obs["has_ice"].shape == (1, 48, 48)
    obs = {
        key: torch.from_numpy(value).float()
        for key, value in obs.items()
    }

    return obs


N_MINIMAP_MAGNIFICATIONS = 4


def get_minimap_obs(full_obs: dict[str, Tensor], pos: Tensor) -> list[Tensor]:
    """
    Create minimaps for a set of features around (x, y).
    Return a list of four minimap magnifications, each with all features concated.
    """
    def _mean_pool(arr: Tensor, window: int) -> Tensor:
        arr = arr.unfold(1, window, window).unfold(2, window, window)
        return torch.mean(arr, dim=(3, 4))

    def _get_minimaps(obs: Tensor, x: Tensor, y: Tensor) -> list[Tensor]:
        n_players = obs.shape[0]
        expanded_map = torch.full((n_players, 96, 96), -1.0)
        # minimap = torch.zeros((n_players * 4, 12, 12))
        for p in range(n_players):
            # unit is in lower right pixel of upper left quadrant
            expanded_map[p][x : x + 48, y : y + 48] = obs[p]
        minimaps = [
            expanded_map[:, 42:54, 42:54],  # small map (12x12 area)
            _mean_pool(expanded_map[:, 36:60, 36:60], 2),  # medium map (24x24 area)
            _mean_pool(expanded_map[:, 24:72, 24:72], 4),  # large map (48x48 area)
            _mean_pool(expanded_map[:], 8),  # full map (96x96 area)
        ]
        # for minimap in minimaps:
            # assert (
                # len(minimap.shape) == 3
                # # variable second dimension (n_players)
                # and minimap.shape[1] == 12
                # and minimap.shape[2] == 12
            # )
        return minimaps

    mini_obs = {
        key: _get_minimaps(value, pos[0], pos[1])
        for key, value in full_obs.items()
    }
    mini_obs = [
        torch.cat([mini_obs[key][i] for key in mini_obs.keys()], dim=0)
        for i in range(4)
    ]
    return mini_obs

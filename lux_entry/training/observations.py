from gym import spaces
import numpy as np
import torch
from torch import Tensor

from luxai_s2.state.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player


per_map_tile_obs_keys = [
    # binary yes/no
    "has_ice",
    "has_ore",
    "player_has_factory",
    "player_has_robot",
    "player_has_light_robot",
    "player_has_heavy_robot",
    "player_has_lichen",
    # normalized from 0-1, -1 means inapplicable
    "rubble",
    "player_lichen",
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
    "player_tot_factories_unb",
    "player_tot_factory_ice_unb",
    "player_tot_factory_ore_unb",
    "player_tot_factory_water_unb",
    "player_tot_factory_metal_unb",
    "player_tot_factory_power_unb",
    "player_tot_lichen_unb",
    # normalized from 0-1
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
    MAX_FS = env_cfg.MAX_FACTORIES
    MAX_RUBBLE = env_cfg.MAX_RUBBLE
    MAX_LICHEN = env_cfg.MAX_LICHEN_PER_TILE
    LIGHT_CARGO_SPACE = env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
    LIGHT_BAT_CAP = env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY
    HEAVY_CARGO_SPACE = env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
    HEAVY_BAT_CAP = env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY
    EXP_MAX_F_CARGO = 2 * HEAVY_CARGO_SPACE
    ICE_WATER_RATIO = env_cfg.ICE_WATER_RATIO
    ORE_METAL_RATIO = env_cfg.ORE_METAL_RATIO
    EXP_MAX_F_WATER = EXP_MAX_F_CARGO / ICE_WATER_RATIO
    EXP_MAX_F_METAL = EXP_MAX_F_CARGO / ORE_METAL_RATIO
    EXP_MAX_TOT_F_CARGO = MAX_FS * EXP_MAX_F_CARGO
    EXP_MAX_TOT_F_WATER = EXP_MAX_TOT_F_CARGO / ICE_WATER_RATIO
    EXP_MAX_TOT_F_METAL = EXP_MAX_TOT_F_CARGO / ORE_METAL_RATIO
    EXP_MAX_F_POWER = env_cfg.FACTORY_CHARGE * 30
    EXP_MAX_TOT_F_POWER = EXP_MAX_F_POWER * MAX_FS
    EXP_MAX_RS = 50
    CYCLE_LENGTH = env_cfg.CYCLE_LENGTH
    DAY_LENGTH = env_cfg.DAY_LENGTH
    MAX_EPISODE_LENGTH = env_cfg.max_episode_length

    # init keys in observation as array of zeros of the right size
    observation_space = get_full_obs_space(env_cfg)
    obs = {
        key: np.zeros(value.shape)
        if type(value) == spaces.MultiBinary
        else np.zeros(value.shape) + value.low[0]
        for key, value in observation_space.items()
    }

    # fill in observations
    obs["has_ice"][0] = env_obs["board"]["ice"]
    obs["has_ore"][0] = env_obs["board"]["ore"]
    obs["rubble"][0] = env_obs["board"]["rubble"] / MAX_RUBBLE
    lichen_strains = [[], []]
    for p, player in enumerate([player, opponent]):
        for f in env_obs["factories"][player].values():
            cargo = f["cargo"]
            p_pos = (p, f["pos"][0], f["pos"][1])
            lichen_strains[p].append(f["strain_id"])
            obs["player_has_factory"][p_pos] = 1
            obs["player_factory_ice_unb"][p_pos] = cargo["ice"] / EXP_MAX_F_CARGO
            obs["player_factory_ore_unb"][p_pos] = cargo["ore"] / EXP_MAX_F_CARGO
            obs["player_factory_water_unb"][p_pos] = cargo["water"] / EXP_MAX_F_WATER
            obs["player_factory_metal_unb"][p_pos] = cargo["metal"] / EXP_MAX_F_METAL
            obs["player_factory_power_unb"][p_pos] = f["power"] / EXP_MAX_F_POWER
            obs["player_tot_factory_ice_unb"][p] += cargo["ice"] / EXP_MAX_TOT_F_CARGO
            obs["player_tot_factory_ore_unb"][p] += cargo["ore"] / EXP_MAX_TOT_F_CARGO
            obs["player_tot_factory_water_unb"][p] += cargo["water"] / EXP_MAX_TOT_F_WATER
            obs["player_tot_factory_metal_unb"][p] += cargo["metal"] / EXP_MAX_TOT_F_METAL
            obs["player_tot_factory_power_unb"][p] += f["power"] / EXP_MAX_TOT_F_POWER
        for r in env_obs["units"][player].values():
            cargo = r["cargo"]
            p_pos = (p, r["pos"][0], r["pos"][1])
            obs["player_has_robot"][p_pos] = 1
            if r["unit_type"] == "LIGHT":
                obs["player_has_light_robot"][p_pos] = 1
                obs["player_robot_power"][p_pos] = r["power"] / LIGHT_BAT_CAP
                obs["player_robot_ice"][p_pos] = cargo["ice"] / LIGHT_CARGO_SPACE
                obs["player_robot_ore"][p_pos] = cargo["ore"] / LIGHT_CARGO_SPACE
                obs["player_robot_cargo"][p_pos] = (cargo["ice"] + cargo["ore"]) / LIGHT_CARGO_SPACE
                obs["player_tot_light_robots_unb"][p] += 1 / EXP_MAX_RS
            elif r["unit_type"] == "HEAVY":
                obs["player_has_heavy_robot"][p_pos] = 1
                obs["player_robot_power"][p_pos] = r["power"] / HEAVY_BAT_CAP
                obs["player_robot_ice"][p_pos] = cargo["ice"] / HEAVY_CARGO_SPACE
                obs["player_robot_ore"][p_pos] = cargo["ore"] / HEAVY_CARGO_SPACE
                obs["player_robot_cargo"][p_pos] = (cargo["ice"] + cargo["ore"]) / HEAVY_CARGO_SPACE
                obs["player_tot_heavy_robots_unb"][p] += 1 / EXP_MAX_RS
        obs["player_tot_robots_unb"][p] += len(env_obs["units"][player]) / EXP_MAX_RS
        obs["player_has_lichen"][p] = env_obs["board"]["lichen_strains"] == lichen_strains[p]
        obs["player_lichen"][p] = (
            env_obs["board"]["lichen"] / MAX_LICHEN
            * (env_obs["board"]["lichen_strains"] == lichen_strains[p])
        )
        obs["player_tot_lichen_unb"][p] += np.sum(obs["player_lichen"][p])
    game_is_day = env_obs["real_env_steps"] % CYCLE_LENGTH < DAY_LENGTH
    obs["game_is_day"][0] += game_is_day
    obs["game_day_or_night_elapsed"][0] += (
        (env_obs["real_env_steps"] % CYCLE_LENGTH) / DAY_LENGTH
        if game_is_day
        else (env_obs["real_env_steps"] % CYCLE_LENGTH - DAY_LENGTH)
        / (CYCLE_LENGTH - DAY_LENGTH)
    )
    obs["game_time_elapsed"][0] += env_obs["real_env_steps"] / MAX_EPISODE_LENGTH
    # assert [k for k in obs.keys()] == obs_keys
    # assert obs["has_ice"].shape == (1, 48, 48)
    obs = {
        key: torch.from_numpy(value).float()
        for key, value in obs.items()
    }

    return obs


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

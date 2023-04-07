from collections import defaultdict
import numpy as np
from typing import Optional

from luxai_s2.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player
from lux_entry.lux.stats import StatsStateDict


def dist(pos1: np.ndarray, pos2: np.ndarray) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def ice_mining_reward(
    obs: ObservationStateDict,
    stats: dict[Player, StatsStateDict],
    player: Player,
    env_cfg: EnvConfig,
    prev_reward_calculations: Optional[dict],
) -> tuple[float, dict, dict]:
    units = obs["units"][player]
    factories = obs["factories"][player]
    factory_positions = {
        factory_id: factory["pos"]
        for factory_id, factory in factories.items()
    }
    ice_positions = np.argwhere(obs["board"]["ice"])
    dist_to_nearest_factory = {
        unit_id: min([
            dist(unit["pos"], factory_pos)
            for factory_pos in factory_positions.values()
        ])
        for unit_id, unit in units.items()
        if len(factory_positions) > 0
    }
    dist_to_nearest_ice = {
        unit_id: min([
            dist(unit["pos"], ice_pos)
            for ice_pos in ice_positions
        ])
        for unit_id, unit in units.items()
        if len(ice_positions) > 0
    }
    ice_in_factory = {
        factory_id: factory["cargo"]["ice"]
        for factory_id, factory in factories.items()
    }
    ice_in_cargo = {
        unit_id: unit["cargo"]["ice"]
        for unit_id, unit in units.items()
    }
    L_CAR_SPACE = env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
    H_CAR_SPACE = env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
    cargo_full = {
        unit_id: (
            total_cargo == L_CAR_SPACE
            if unit["unit_type"] == "LIGHT"
            else total_cargo == H_CAR_SPACE
        )
        for unit_id, unit in units.items()
        if (total_cargo := unit["cargo"]["ice"] + unit["cargo"]["ore"])
    }

    my_stats = stats[player]
    metrics: dict[str, float] = {
        "total_ice_dug": (
            my_stats["generation"]["ice"]["HEAVY"] + my_stats["generation"]["ice"]["LIGHT"]
        ),
        "total_water_produced": my_stats["generation"]["water"],
        "action_queue_updates_success": my_stats["action_queue_updates_success"],
        "action_queue_updates_total": my_stats["action_queue_updates_total"],
    }

    reward = 0
    if prev_reward_calculations is not None:
        delta_dist_to_nearest_factory = {
            unit_id: (
                prev_reward_calculations["dist_to_nearest_factory"][unit_id] - dist
                if unit_id in prev_reward_calculations["dist_to_nearest_factory"]
                else 0
            )
            for unit_id, dist in dist_to_nearest_factory.items()
        }
        metric = list(delta_dist_to_nearest_factory.values())
        if len(metric) > 0:
            metrics["avg_delta_dist_to_nearest_factory"] = sum(metric) / len(metric)
        delta_dist_to_nearest_ice = {
            unit_id: (
                prev_reward_calculations["dist_to_nearest_ice"][unit_id] - dist
                if unit_id in prev_reward_calculations["dist_to_nearest_ice"]
                else 0
            )
            for unit_id, dist in dist_to_nearest_ice.items()
        }
        metric = list(delta_dist_to_nearest_ice.values())
        if len(metric) > 0:
            metrics["avg_delta_dist_to_nearest_ice"] = sum(metric) / len(metric)
        delta_ice_in_factory = {
            factory_id: (
                ice - prev_reward_calculations["ice_in_factory"][factory_id]
                if factory_id in prev_reward_calculations["ice_in_factory"]
                else ice
            )
            for factory_id, ice in ice_in_factory.items()
        }
        metrics["delta_ice_in_factory"] = sum(list(delta_ice_in_factory.values()))
        ice_delivered = defaultdict(int)
        for factory_id, delta_ice in delta_ice_in_factory.items():
            if delta_ice == 0:
                continue
            for unit_id, unit in units.items():
                if unit["pos"] == factory_positions[factory_id]:
                    ice_delivered[unit_id] = delta_ice
        metrics["ice_delivered"] = sum(list(ice_delivered.values()))
        ice_mined = {
            unit_id: (
                ice - prev_reward_calculations["ice_in_cargo"][unit_id]
                if unit_id in prev_reward_calculations["ice_in_cargo"]
                else ice
            )
            for unit_id, ice in ice_in_cargo.items()
        }
        reward = sum([
            delta_dist_to_nearest_factory[unit_id] + ice_delivered[unit_id]
            if unit_id in prev_reward_calculations["cargo_full"]
            and prev_reward_calculations["cargo_full"][unit_id]
            else delta_dist_to_nearest_ice[unit_id] + ice_mined[unit_id]
            for unit_id in units.keys()
        ])
        # print(
            # "dist_to_nearest_factory:", dist_to_nearest_factory, "\n",
            # "dist_to_nearest_ice:", dist_to_nearest_ice, "\n",
            # "delta_dist_to_nearest_factory:", delta_dist_to_nearest_factory, "\n",
            # "delta_dist_to_nearest_ice:", delta_dist_to_nearest_ice, "\n",
            # "delta_ice_in_factory:", delta_ice_in_factory, "\n",
            # "ice_delivered:", ice_delivered, "\n",
            # "ice_mined:", ice_mined, "\n",
            # "reward:", reward, "\n",
            # "metrics:", metrics, "\n",
        # )
    prev_reward_calculations = {
        "dist_to_nearest_factory": dist_to_nearest_factory,
        "dist_to_nearest_ice": dist_to_nearest_ice,
        "ice_in_factory": ice_in_factory,
        "ice_in_cargo": ice_in_cargo,
        "cargo_full": cargo_full,
    }

    return reward, prev_reward_calculations, metrics

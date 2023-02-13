import numpy as np

from luxai_s2.unit import FactoryPlacementActionType
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice, random_factory_placement
from luxai_s2.state import ObservationStateDict


def factory_placement_policy(player, obs: ObservationStateDict) -> FactoryPlacementActionType:
    potential_spawns = np.array(
        list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    )
    spawn_loc = potential_spawns[
        np.random.randint(0, len(potential_spawns))
    ]
    return FactoryPlacementActionType(spawn=spawn_loc, metal=150, water=150)

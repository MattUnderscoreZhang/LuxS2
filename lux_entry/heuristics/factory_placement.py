from typing import TypedDict
import numpy as np

from luxai_s2.state import ObservationStateDict

from lux_entry.lux.state import Player


class FactoryPlacementActionType(TypedDict):
    metal: int
    water: int
    spawn: np.ndarray


def random_factory_placement(
    player: Player, obs: ObservationStateDict
) -> FactoryPlacementActionType:
    potential_spawns = np.array(
        list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    )
    spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
    metal = obs["teams"][player]["metal"]
    water = obs["teams"][player]["water"]
    factories_to_place = obs["teams"][player]["factories_to_place"]
    return FactoryPlacementActionType(
        spawn=spawn_loc, metal=metal//factories_to_place, water=water//factories_to_place,
    )


# TODO: game fps shifts between >3000 and ~400 depending on very small changes here - figure out why
def place_near_random_ice(
    player: Player, obs: ObservationStateDict
) -> FactoryPlacementActionType:
    potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    potential_spawns_set = set(potential_spawns)
    # find all places to the left of a block of ice
    pot_ice_spots = np.argwhere(np.diff(obs["board"]["ice"]) == 1)
    if len(pot_ice_spots) == 0:
        pot_ice_spots = potential_spawns
    factory_size = 3
    pos = None
    done_search = False
    for _ in range(5):
        pos = pot_ice_spots[np.random.randint(0, len(pot_ice_spots))]
        assert len(pos) == 2
        for x in range(factory_size):
            for y in range(factory_size):
                check_pos = [
                    pos[0] + x - factory_size // 2,
                    pos[1] + y - factory_size // 2,
                ]
                if tuple(check_pos) in potential_spawns_set:
                    done_search = True
                    pos = check_pos
                    break
            if done_search:
                break
        if done_search:
            break
    pos = (
        np.array(potential_spawns[np.random.randint(0, len(potential_spawns))])
        if not done_search
        else np.array(pos)
    )
    metal = obs["teams"][player]["metal"]
    water = obs["teams"][player]["water"]
    factories_to_place = obs["teams"][player]["factories_to_place"]
    return FactoryPlacementActionType(
        spawn=pos, metal=metal//factories_to_place, water=water//factories_to_place
    )

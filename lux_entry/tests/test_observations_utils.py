import pytest

from lux_entry.components.observations import custom_observations, utils


@pytest.fixture
def full_obs() -> custom_observations.FullObservation:
    ...

def test_get_full_obs():
    # TODO: write test
    ...


def test_get_partial_obs_around_unit():
    # TODO: write test
    ...


def test_construct_obs(obs_dict: custom_observations.Observation):
    # TODO: write test
    pass
    # test_convert_obs_to_tensor(all_observables: list[np.ndarray], pass_through_observables: list[np.ndarray])
    all_observables = [
        # [binary yes/no]
        obs_dict.tile_has_ice, # obs_dict.tile_has_ore,
        # obs_dict.tile_has_lichen_strain,
        # obs_dict.tile_per_player_has_factory,
        # obs_dict.tile_per_player_has_robot,
        # obs_dict.tile_per_player_has_light_robot,
        # obs_dict.tile_per_player_has_heavy_robot,
        # obs_dict.tile_per_player_has_lichen,
        # # [obs_dict.normalized from 0-1, -1 means inapplicable]
        # obs_dict.tile_rubble,
        # obs_dict.tile_per_player_lichen,
        # obs_dict.tile_per_player_light_robot_power,
        # obs_dict.tile_per_player_light_robot_ice,
        # obs_dict.tile_per_player_light_robot_ore,
        # obs_dict.tile_per_player_heavy_robot_power,
        # obs_dict.tile_per_player_heavy_robot_ice,
        # obs_dict.tile_per_player_heavy_robot_ore,
        # # [obs_dict.normalized and positive unbounded, -1 means inapplicable]
        # obs_dict.tile_per_player_factory_ice_unbounded,
        # obs_dict.tile_per_player_factory_ore_unbounded,
        # obs_dict.tile_per_player_factory_water_unbounded,
        # obs_dict.tile_per_player_factory_metal_unbounded,
        # obs_dict.tile_per_player_factory_power_unbounded,
        # # [[obs_dict.broadcast features]]
        # # obs_dict.normalized and positive unbounded
        # obs_dict.total_per_player_robots_unbounded,
        # obs_dict.total_per_player_light_robots_unbounded,
        # obs_dict.total_per_player_heavy_robots_unbounded,
        # obs_dict.total_per_player_factories_unbounded,
        # obs_dict.total_per_player_factory_ice_unbounded,
        # obs_dict.total_per_player_factory_ore_unbounded,
        # obs_dict.total_per_player_factory_water_unbounded,
        # obs_dict.total_per_player_factory_metal_unbounded,
        # obs_dict.total_per_player_factory_power_unbounded,
        # obs_dict.total_per_player_lichen_unbounded,
        # # obs_dict.normalized from 0-1
        # obs_dict.game_is_day,
        # obs_dict.game_day_or_night_elapsed,
        # obs_dict.game_time_elapsed,
        # # [[obs_dict.bidding and factory placement info]]
        # obs_dict.teams,
        # obs_dict.factories_per_team,
        # obs_dict.valid_spawns_mask,
    ]
    pass_through_observables = [
        obs_dict.tile_has_ice, # obs_dict.tile_has_ore,
    ]

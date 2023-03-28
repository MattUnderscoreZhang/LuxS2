# from dataclasses import dataclass


# @dataclass
# class TrainingEnvironment:
    # name: str
    # description: str
    # bid_policy: BidPolicy
    # factory_placement_policy: FactoryPlacementPolicy
    # controller: Controller
    # get_conv_obs: GetObservables
    # get_skip_obs: GetObservables
    # get_actions_mask: GetActionsMask


# def get_min_conv_obs():
    # obs_dict.tile_has_ice,
    # ...


# def get_max_conv_obs(obs:dict: Observation):
    # return [
        # obs_dict.tile_has_ice,
        # obs_dict.tile_per_player_has_factory,
        # obs_dict.tile_per_player_has_robot,
        # obs_dict.tile_per_player_has_light_robot,
        # obs_dict.tile_per_player_has_heavy_robot,
        # obs_dict.tile_rubble,
        # obs_dict.tile_per_player_light_robot_power,
        # obs_dict.tile_per_player_heavy_robot_power,
        # obs_dict.tile_per_player_factory_ice_unbounded,
        # obs_dict.tile_per_player_factory_ore_unbounded,
        # obs_dict.tile_per_player_factory_water_unbounded,
        # obs_dict.tile_per_player_factory_metal_unbounded,
        # obs_dict.tile_per_player_factory_power_unbounded,
        # obs_dict.game_is_day,
        # obs_dict.game_day_or_night_elapsed,
    # ]


# def get_skip_obs():
    # obs_dict.tile_has_ice,
    # obs_dict.tile_per_player_has_factory,
    # ...


# def get_actions_mask():
    # ...


# environments = [
    # TrainingEnvironment(
        # name="ice_crawler",
        # description="Single random weight robot mines nearby ice without clearing rubble, one action at a time.",
        # bid_policy=zero_bid,
        # factory_placement_policy=place_near_random_ice,
        # controller=single_unit_controller,  # TODO: research action controllers
        # get_conv_obs=get_min_conv_obs,
        # get_skip_obs=get_skip_obs,
        # get_actions_mask=get_actions_mask,  # TODO: research actions
    # ),
    # TrainingEnvironment(
        # name="ice_walker",
        # description="Single random weight robot mines ice anywhere without clearing rubble, one action at a time.",
        # bid_policy=zero_bid,
        # factory_placement_policy=place_far_from_ice,
        # controller=single_unit_controller,  # TODO: research action controllers
        # get_conv_obs=get_min_conv_obs,
        # get_skip_obs=get_skip_obs,
        # get_actions_mask=get_actions_mask,  # TODO: research actions
    # ),
    # TrainingEnvironment(
        # name="ice_runner",
        # description="Single random weight robot mines ice anywhere while clearing rubble and picking up power, one action at a time.",
        # bid_policy=zero_bid,
        # factory_placement_policy=place_far_from_ice,
        # controller=single_unit_controller,  # TODO: research action controllers
        # get_conv_obs=get_max_conv_obs,
        # get_skip_obs=get_skip_obs,
        # get_actions_mask=get_actions_mask,  # TODO: research actions
    # ),
    # TrainingEnvironment(
        # name="ice_glider",
        # description="Single random weight robot mines ice, using action sequences.",
        # bid_policy=zero_bid,
        # factory_placement_policy=place_far_from_ice,
        # controller=single_unit_controller,  # TODO: research action controllers
        # get_conv_obs=get_max_conv_obs,
        # get_skip_obs=get_skip_obs,
        # get_actions_mask=get_actions_mask,  # TODO: research actions
    # ),
    # # team training environments
    # # TODO: have a way to specify:
    # # - any number of agents for each player
    # # - each agent using an independent behavior
    # # - any number training, any number with fixed weights

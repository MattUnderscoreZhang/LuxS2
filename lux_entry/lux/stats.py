from typing import TypedDict


class RobotStatsStateDict(TypedDict):
    LIGHT: int
    HEAVY: int


def create_robot_stats() -> RobotStatsStateDict:
    return RobotStatsStateDict(LIGHT=0, HEAVY=0)


class AllStatsStateDict(TypedDict):
    LIGHT: int
    HEAVY: int
    FACTORY: int


def create_all_stats() -> AllStatsStateDict:
    return AllStatsStateDict(LIGHT=0, HEAVY=0, FACTORY=0)


class GenerationStatsStateDict(TypedDict):
    power: AllStatsStateDict
    water: int
    metal: int
    ore: RobotStatsStateDict  # amount dug out by HEAVY or LIGHT
    ice: RobotStatsStateDict  # amount dug out by HEAVY or LIGHT
    lichen: int  # amount grown
    built: RobotStatsStateDict  # amount built


def create_generation_stats() -> GenerationStatsStateDict:
    return GenerationStatsStateDict(
        power=create_all_stats(),
        water=0,
        metal=0,
        ore=create_robot_stats(),
        ice=create_robot_stats(),
        lichen=0,
        built=create_robot_stats(),
    )


class ConsumptionStatsStateDict(TypedDict):
    power: AllStatsStateDict
    water: int
    metal: int


def create_consumption_stats() -> ConsumptionStatsStateDict:
    return ConsumptionStatsStateDict(
        power=create_all_stats(),
        water=0,
        metal=0,
    )


class TransferStatsStateDict(TypedDict):
    power: int
    water: int
    metal: int
    ice: int
    ore: int


def create_transfer_pickup_stats():
    return TransferStatsStateDict(power=0, water=0, metal=0, ice=0, ore=0)


class PickUpStatsStateDict(TypedDict):
    power: int
    water: int
    metal: int
    ice: int
    ore: int


class DestroyedStatsStateDict(TypedDict):
    FACTORY: int
    HEAVY: int
    LIGHT: int
    rubble: RobotStatsStateDict
    lichen: RobotStatsStateDict


def create_destroyed_stats():
    return DestroyedStatsStateDict(
        FACTORY=0,
        HEAVY=0,
        LIGHT=0,
        rubble=create_robot_stats(),
        lichen=create_robot_stats(),
    )


class StatsStateDict(TypedDict):
    consumption: ConsumptionStatsStateDict
    generation: GenerationStatsStateDict
    action_queue_updates_success: int
    action_queue_updates_total: int
    destroyed: DestroyedStatsStateDict
    transfer: TransferStatsStateDict
    pickup: PickUpStatsStateDict


def create_empty_stats() -> StatsStateDict:
    return StatsStateDict(
        action_queue_updates_total=0,
        action_queue_updates_success=0,
        consumption=create_consumption_stats(),
        destroyed=create_destroyed_stats(),
        generation=create_generation_stats(),
        pickup=create_transfer_pickup_stats(),
        transfer=create_transfer_pickup_stats(),
    )

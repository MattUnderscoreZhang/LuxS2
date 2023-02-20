from typing import TypedDict

from luxai_s2.state import ObservationStateDict

from lux_entry.lux.state import Player


class BidActionType(TypedDict):
    faction: str
    bid: int


def zero_bid(player: Player, obs: ObservationStateDict) -> BidActionType:
    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"
    return BidActionType(bid=0, faction=faction)

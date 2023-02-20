from typing import TypedDict

from luxai_s2.state import ObservationStateDict


class BidActionType(TypedDict):
    faction: str
    bid: int


def zero_bid(player, obs: ObservationStateDict) -> BidActionType:
    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"
    return BidActionType(bid=0, faction=faction)

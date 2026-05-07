import random
from typing import Tuple
from phyelds.libraries.device import local_id
from phyelds.data import NeighborhoodField, StateT
from phyelds.libraries.spreading import distance_to
from phyelds.libraries.utils import min_with_default
from phyelds.calculus import aggregate, remember, neighbors
from phyelds.libraries.leader_election import random_uuid, breaking_using_uids

@aggregate
def elect_leaders(area: float, distances: NeighborhoodField[float]) -> Tuple[bool, int]:
    result: Tuple[StateT[float], int] = breaking_using_uids(random_uuid(), area, distances)
    am_i_leader = result[1] == local_id() and result[0] != (float("inf"))
    return am_i_leader, result[1]
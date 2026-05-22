import numpy as np
from mahjong.shanten import Shanten

NUM_TILE_TYPES = 34
MAX_TURNS = 18
MAX_SHANTEN = 8

AKA_5M, AKA_5P, AKA_5S = 16, 52, 88
FEATURE_DIM = 187

_shanten = Shanten()

def tile_to_34(tile136: int) -> int:
    return tile136 // 4

def tiles_to_34_array(tile_ids: list[int]) -> np.ndarray:
    arr = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for tile in tile_ids:
        if tile >= 0:
            arr[tile_to_34(tile)] += 1.0
    return arr

def compute_shanten(hand_136: list[int]) -> int:
    hand_34 = [0] * 34
    for tile in hand_136:
        hand_34[tile_to_34(tile)] += 1
    
    tile_count = sum(hand_34)
    # Added 13 to handle off-turn calling states safely
    if tile_count not in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]:
        return MAX_SHANTEN

    return _shanten.calculate_shanten(hand_34)

def encode_wind(wind: int) -> np.ndarray:
    arr = np.zeros(4, dtype=np.float32)
    if 0 <= wind < 4:
        arr[wind] = 1.0
    return arr

def encode_aka_dora(hand_136: list[int]) -> np.ndarray:
    hand_set = set(hand_136)
    return np.array([
        1.0 if AKA_5M in hand_set else 0.0,
        1.0 if AKA_5P in hand_set else 0.0,
        1.0 if AKA_5S in hand_set else 0.0,
    ], dtype=np.float32)

def encode_opponent_discards(record: dict) -> list[np.ndarray]:
    opponents = []
    player_wind = record["player_wind"]

    for offset in [1, 2, 3]:
        opp = (player_wind + offset) % 4
        # Fallback to empty list if player sub-dict is missing keys
        discards = record.get(str(opp), {}).get("discards", [])
        opponents.append(tiles_to_34_array(discards))
    return opponents

def encode_opponent_riichi(record: dict) -> np.ndarray:
    player_wind = record["player_wind"]
    flags = []
    for offset in [1, 2, 3]:
        opp = (player_wind + offset) % 4
        is_riichi = record.get(str(opp), {}).get("riichi", False)
        flags.append(1.0 if is_riichi else 0.0)
    return np.array(flags, dtype=np.float32)

def encode_state_vector(record: dict) -> np.ndarray:
    """
    Purely encodes the 187-dimensional board state.
    Independent of target labels or action types.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    hand = record["hand_tiles"]

    vec[0:34] = tiles_to_34_array(hand)
    vec[34:68] = tiles_to_34_array(record.get("dora_indicators", []))
    vec[68:72] = encode_wind(record["player_wind"])
    vec[72:76] = encode_wind(min(record.get("round_wind", 0) // 4, 3))
    
    remain_tiles = record.get("remain_tiles", 70)
    approx_turn = (70 - remain_tiles) / 4
    vec[76] = min(approx_turn / MAX_TURNS, 1.0)

    shanten = compute_shanten(hand)
    vec[77] = max(shanten, 0) / MAX_SHANTEN
    vec[78] = 1.0 if shanten == 0 else 0.0
    vec[79:82] = encode_aka_dora(hand)

    opp_discards = encode_opponent_discards(record)
    vec[82:116] = opp_discards[0]
    vec[116:150] = opp_discards[1]
    vec[150:184] = opp_discards[2]
    vec[184:187] = encode_opponent_riichi(record)

    return vec
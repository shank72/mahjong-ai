import numpy as np
from mahjong.shanten import Shanten

# constants
NUM_TILE_TYPES = 34

MAX_TURNS = 18
MAX_SHANTEN = 8

AKA_5M = 16
AKA_5P = 52
AKA_5S = 88

FEATURE_DIM = 187

_shanten = Shanten()


# tile helpers
def tile_to_34(tile136: int) -> int:
    return tile136 // 4


def tiles_to_34_array(tile_ids: list[int]) -> np.ndarray:
    """
    Convert list of 136-tile IDs into 34-count vector.
    """
    arr = np.zeros(NUM_TILE_TYPES, dtype=np.float32)

    for tile in tile_ids:
        if tile >= 0:
            arr[tile_to_34(tile)] += 1.0

    return arr


# shanten
def compute_shanten(hand_136: list[int]) -> int:
    """
    Compute shanten from hand.
    """

    hand_34 = [0] * 34

    for tile in hand_136:
        hand_34[tile_to_34(tile)] += 1

    tile_count = sum(hand_34)

    # library only accepts:
    # 1,2,4,5,7,8,10,11,13,14

    if tile_count not in [1,2,4,5,7,8,10,11,13,14]:
        return MAX_SHANTEN

    return _shanten.calculate_shanten(hand_34)


# encoding helpers
def encode_wind(wind: int) -> np.ndarray:
    """
    One-hot wind encoding.
    """
    arr = np.zeros(4, dtype=np.float32)

    if 0 <= wind < 4:
        arr[wind] = 1.0

    return arr


def encode_aka_dora(hand_136: list[int]) -> np.ndarray:
    """
    [red5m, red5p, red5s]
    """

    hand_set = set(hand_136)

    return np.array([
        1.0 if AKA_5M in hand_set else 0.0,
        1.0 if AKA_5P in hand_set else 0.0,
        1.0 if AKA_5S in hand_set else 0.0,
    ], dtype=np.float32)


def encode_opponent_discards(record: dict) -> list[np.ndarray]:
    """
    Encode opponents' visible discards.

    Kaggle format:
        record["0"]["discards"]
        record["1"]["discards"]
        etc.
    """

    opponents = []

    player_wind = record["player_wind"]

    # start at current player and go clockwise through opponents
    for offset in [1, 2, 3]:
        opp = (player_wind + offset) % 4

        discards = record[str(opp)]["discards"]

        opponents.append(
            tiles_to_34_array(discards)
        )

    return opponents


def encode_opponent_riichi(record: dict) -> np.ndarray:
    """
    Encode opponent riichi flags.
    """

    player_wind = record["player_wind"]

    flags = []

    # start at current player and go clockwise through opponents
    for offset in [1, 2, 3]:
        opp = (player_wind + offset) % 4

        flags.append(
            1.0 if record[str(opp)]["riichi"] else 0.0
        )

    return np.array(flags, dtype=np.float32)


# main encoder
def encode(record: dict):
    """
    Convert raw dataset record into:
        x = feature vector
        y = discard label (0-33)
    """

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    hand = record["hand_tiles"]

    # hand
    vec[0:34] = tiles_to_34_array(hand)

    # dora indicators
    vec[34:68] = tiles_to_34_array(
        record["dora_indicators"]
    )

    # seat wind
    vec[68:72] = encode_wind(
        record["player_wind"]
    )

    # round wind
    round_wind = record["round_wind"] // 4

    vec[72:76] = encode_wind(
        min(round_wind, 3)
    )

    # turn number
    remain_tiles = record["remain_tiles"]

    approx_turn = (70 - remain_tiles) / 4

    vec[76] = min(approx_turn / MAX_TURNS, 1.0)

    # shanten
    shanten = compute_shanten(hand)
    vec[77] = max(shanten, 0) / MAX_SHANTEN

    # is the hand in tenpai?
    vec[78] = 1.0 if shanten == 0 else 0.0

    # aka dora
    vec[79:82] = encode_aka_dora(hand)

    # opponent discards
    opp_discards = encode_opponent_discards(record)

    vec[82:116] = opp_discards[0]
    vec[116:150] = opp_discards[1]
    vec[150:184] = opp_discards[2]

    # opponent riichi
    vec[184:187] = encode_opponent_riichi(record)

    # label
    action_idx = record["action_idx"]

    action = record["valid_actions"][action_idx]

    discard_tile = action["tiles"][0]

    label = tile_to_34(discard_tile)

    return vec, label


# batch encoder
def encode_batch(records: list[dict]):
    X = []
    y = []

    for record in records:
        vec, label = encode(record)

        X.append(vec)
        y.append(label)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64)
    )
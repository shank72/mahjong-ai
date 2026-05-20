"""
data/feature_encoder.py

Converts a parsed game state dict (output of parse_tenhou.py) into a
fixed-length numpy feature vector for model training and inference.

INFORMATION BOUNDARY — only information a real player could see:

  Y  Your own 14-tile hand (before discard)
  Y  Dora indicators (publicly visible)
  Y  Seat wind, round wind, turn number
  Y  Shanten number (computable from your own hand)
  Y  Aka Dora flags (you can see your own red tiles)
  Y  Each opponent's visible discard pile
  Y  Each opponent's riichi status

  N  Opponent hand contents  — never included
  N  Opponent drawn tile IDs — never included

FEATURE VECTOR LAYOUT  (total: 187 elements)
  [0:34]    hand_34          tile type counts in your hand
  [34:68]   dora_34          tile type counts of dora indicators
  [68:72]   seat_wind        one-hot (East/South/West/North)
  [72:76]   round_wind       one-hot (East/South/West/North) - only E/S used in riichi
  [76]      turn_norm        turn number normalized 0-1 (max 18 draws)
  [77]      shanten          shanten number normalized (raw / 8.0)
  [78]      is_tenpai        1.0 if shanten == 0 else 0.0
  [79:82]   aka_dora         red 5m / 5p / 5s in your hand (binary flags)
  [82:116]  opp1_discards    34-count array of opponent 1's visible discards
  [116:150] opp2_discards    34-count array of opponent 2's visible discards
  [150:184] opp3_discards    34-count array of opponent 3's visible discards
  [184]     opp1_riichi      binary
  [185]     opp2_riichi      binary
  [186]     opp3_riichi      binary

TARGET (label):
  discard_34: int in [0, 33] — the tile TYPE discarded (tile_id // 4)
  This is the 34-class classification target.
"""

import numpy as np
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

# ── constants ────────────────────────────────────────────────────────────────

NUM_TILE_TYPES  = 34      # unique tile types in Riichi Mahjong
MAX_TURNS       = 18      # maximum draws per player in a round (approx)
MAX_SHANTEN     = 8       # maximum shanten for a 13-tile hand - smaller = better

# Tenhou tile IDs for aka dora (red fives)
AKA_5M = 16
AKA_5P = 52
AKA_5S = 88
AKA_DORA_IDS = {AKA_5M, AKA_5P, AKA_5S}

# Feature vector slice indices — edit here if you add/remove features
SLICE_HAND        = (0,   34)
SLICE_DORA        = (34,  68)
SLICE_SEAT_WIND   = (68,  72)
SLICE_ROUND_WIND  = (72,  76)
IDX_TURN_NORM     = 76
IDX_SHANTEN       = 77
IDX_IS_TENPAI     = 78
SLICE_AKA         = (79,  82)
SLICE_OPP1        = (82,  116)
SLICE_OPP2        = (116, 150)
SLICE_OPP3        = (150, 184)
IDX_OPP1_RIICHI   = 184
IDX_OPP2_RIICHI   = 185
IDX_OPP3_RIICHI   = 186

FEATURE_DIM = 187    # total feature vector length

# shanten calculator (singleton — expensive to reinstantiate)
_shanten_calc = Shanten()


# tile conversion helpers 
def tile_to_34(tile_id: int) -> int:
    """
    Collapse a 136-encoding tile ID to a 34-type index.
    All 4 physical copies of a tile type map to the same index:
      tile_id 0,1,2,3   → type 0  (1-man)
      tile_id 4,5,6,7   → type 1  (2-man)
      ...
      tile_id 16,17,18,19 → type 4  (5-man, including aka dora at id=16)
      ...
      tile_id 132-135   → type 33 (chun / red dragon)
    """
    return tile_id // 4


def hand_136_to_34_array(hand_136: list[int]) -> np.ndarray:
    """
    Convert a list of 136-encoding tile IDs into a 34-length count array.

    Example:
        hand_136 = [0, 4, 16, 20, 20, 68]  # 1m 2m 5m 6m 6m 9p
        → count_34[0]=1, count_34[1]=1, count_34[4]=1,
          count_34[5]=2, count_34[17]=1  (all others 0)

    This is the format expected by mahjong.Shanten and by the model.
    Red fives (tile_id 16/52/88) land on the correct type index (4/13/22)
    but their aka-dora status is captured separately in encode_aka_dora().
    """
    arr = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for tile_id in hand_136:
        arr[tile_to_34(tile_id)] += 1.0
    return arr


def tiles_to_34_array(tile_ids: list[int]) -> np.ndarray:
    """Same as hand_136_to_34_array but for any list of tile IDs (e.g. discards, dora)."""
    arr = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for tile_id in tile_ids:
        arr[tile_to_34(tile_id)] += 1.0
    return arr


# ── shanten helper ────────────────────────────────────────────────────────────

def compute_shanten(hand_136: list[int]) -> int:
    """
    Compute shanten number for a hand using the mahjong library.

    Shanten = tiles away from tenpai.
      -1 = complete winning hand
       0 = tenpai (one tile away from winning)
       1 = one tile away from tenpai
       ...
       8 = maximum (completely unformed hand)

    The library expects a 34-length integer count array, not 136 IDs.
    We take the first 13 tiles if 14 are present (library works on 13-tile hands).
    """
    # Use first 13 tiles for shanten calc (before the draw is decided on)
    tiles_for_calc = hand_136[:13]
    hand_34_int = [0] * NUM_TILE_TYPES
    for tile_id in tiles_for_calc:
        hand_34_int[tile_to_34(tile_id)] += 1
    return _shanten_calc.calculate_shanten(hand_34_int)


# ── per-feature encoders ──────────────────────────────────────────────────────

def encode_hand(hand_136: list[int]) -> np.ndarray:
    """34-length count array of tile types in the acting player's hand."""
    return hand_136_to_34_array(hand_136)


def encode_dora(dora_indicators: list[int]) -> np.ndarray:
    """
    34-length count array of dora indicator tile types.

    NOTE: the dora itself is the tile AFTER the indicator in the sequence:
      1m indicator → 2m is dora
      9m indicator → 1m is dora (wraps)
      North indicator → East is dora (winds wrap separately)
    We encode the raw indicators here; the model learns the +1 mapping implicitly
    from training data. You could also pre-shift to actual dora tiles — either works.
    """
    return tiles_to_34_array(dora_indicators)


def encode_wind(wind_idx: int, size: int = 4) -> np.ndarray:
    """One-hot encode a wind index. size=4 for seat wind, size=2 for round wind."""
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= wind_idx < size:
        arr[wind_idx] = 1.0
    return arr


def encode_aka_dora(hand_136: list[int]) -> np.ndarray:
    """
    3-element binary array: [has_red_5m, has_red_5p, has_red_5s].

    This captures the information lost when collapsing 136 → 34.
    Red fives are worth an extra han, so the model needs to know about them
    separately from the plain tile count.

    Example:
        hand contains tile_id 16 (aka 5m) and tile_id 17 (normal 5m)
        hand_34[4] = 2  (two 5-mans total — loses which is red)
        aka_flags  = [1, 0, 0]  (has red 5m — preserved here)
    """
    hand_set = set(hand_136)
    return np.array([
        1.0 if AKA_5M in hand_set else 0.0,
        1.0 if AKA_5P in hand_set else 0.0,
        1.0 if AKA_5S in hand_set else 0.0,
    ], dtype=np.float32)


def encode_opponent_discards(opponent_discards: list[list[int]]) -> list[np.ndarray]:
    """
    Encode each opponent's visible discard pile as a 34-length count array.

    opponent_discards is a list of 3 lists (one per opponent, in relative seat order).
    Returns a list of 3 arrays, each shape (34,).

    Only uses tile TYPE (tile_id // 4) — never exposes which physical copy was
    discarded. This is identical to what a human player observes on the table.
    """
    result = []
    for discards in opponent_discards:
        result.append(tiles_to_34_array(discards))
    # pad to always return exactly 3 arrays (in case of <3 opponents)
    while len(result) < 3:
        result.append(np.zeros(NUM_TILE_TYPES, dtype=np.float32))
    return result[:3]


# ── main encoder ─────────────────────────────────────────────────────────────

def encode(record: dict) -> tuple[np.ndarray, int]:
    """
    Encode a single parsed game state record into a feature vector and label.

    Args:
        record: dict from parse_tenhou.py with keys:
                hand, discard, dora_indicators, seat_wind, round_wind,
                turn, opponent_discards, opponent_riichi

    Returns:
        (feature_vec, label) where:
            feature_vec: np.ndarray shape (FEATURE_DIM,) = (187,)
            label:       int in [0, 33] — tile type the player discarded
    """
    hand_136        = record["hand"]               # 14 tile IDs
    discard_136     = record["discard"]            # 1 tile ID (the label)
    dora_indicators = record["dora_indicators"]
    seat_wind       = record["seat_wind"]          # 0=E 1=S 2=W 3=N
    round_wind      = record["round_wind"]         # 0=E 1=S
    turn            = record["turn"]               # 0-indexed
    opp_discards    = record["opponent_discards"]  # [[tile_ids], [tile_ids], [tile_ids]]
    opp_riichi      = record["opponent_riichi"]    # [bool, bool, bool]

    # compute shanten from 13-tile hand (before the drawn tile is accounted for)
    shanten = compute_shanten(hand_136)

    # build feature vector
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # hand tile counts
    s, e = SLICE_HAND
    vec[s:e] = encode_hand(hand_136)

    # dora indicator counts
    s, e = SLICE_DORA
    vec[s:e] = encode_dora(dora_indicators)

    # seat wind (one-hot, size 4)
    s, e = SLICE_SEAT_WIND
    vec[s:e] = encode_wind(seat_wind, size=4)

    # round wind (one-hot, size 4 — East/South/West/North but only E/S used in riichi)
    s, e = SLICE_ROUND_WIND
    vec[s:e] = encode_wind(round_wind, size=4)

    # turn number normalized to [0, 1]
    vec[IDX_TURN_NORM] = min(turn / MAX_TURNS, 1.0)

    # shanten normalized to [0, 1]  (-1 stays negative but that means winning hand -> rare)
    vec[IDX_SHANTEN] = max(shanten, 0) / MAX_SHANTEN

    # tenpai binary flag
    vec[IDX_IS_TENPAI] = 1.0 if shanten == 0 else 0.0

    # aka dora flags
    s, e = SLICE_AKA
    vec[s:e] = encode_aka_dora(hand_136)

    # opponent discard counts (tile types only — no hidden info)
    opp_arrays = encode_opponent_discards(opp_discards)
    vec[SLICE_OPP1[0]:SLICE_OPP1[1]] = opp_arrays[0]
    vec[SLICE_OPP2[0]:SLICE_OPP2[1]] = opp_arrays[1]
    vec[SLICE_OPP3[0]:SLICE_OPP3[1]] = opp_arrays[2]

    # opponent riichi flags
    vec[IDX_OPP1_RIICHI] = 1.0 if opp_riichi[0] else 0.0
    vec[IDX_OPP2_RIICHI] = 1.0 if opp_riichi[1] else 0.0
    vec[IDX_OPP3_RIICHI] = 1.0 if opp_riichi[2] else 0.0

    # -- label: tile TYPE discarded (0-33) --
    label = tile_to_34(discard_136)

    return vec, label


def encode_batch(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a list of records into a batch of feature vectors and labels.

    Returns:
        X: np.ndarray shape (N, FEATURE_DIM)
        y: np.ndarray shape (N,) dtype int64
    """
    X, y = [], []
    for record in records:
        vec, label = encode(record)
        X.append(vec)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fake record to verify shapes and values
    dummy_record = {
        "hand":              [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
        #                     1m 2m 3m 4m  5m* 6m  7m  8m  9m  1p  2p  3p  4p  5p*
        "discard":           52,   # discard red 5-pin
        "dora_indicators":   [4],  # 2m indicator → 3m is dora
        "seat_wind":         0,    # East
        "round_wind":        0,    # East round
        "turn":              3,
        "opponent_discards": [[108, 112], [72, 76, 80], []],
        "opponent_riichi":   [False, True, False],
    }

    vec, label = encode(dummy_record)

    print(f"Feature vector shape : {vec.shape}")           # (187,)
    print(f"Label (tile type)    : {label}")               # 13 (5-pin type index)
    print(f"Hand counts (first 9): {vec[0:9]}")            # man tiles
    print(f"Shanten              : {vec[IDX_SHANTEN]:.3f}")
    print(f"Is tenpai            : {vec[IDX_IS_TENPAI]}")
    print(f"Aka dora flags       : {vec[79:82]}")          # [1, 1, 0] — has red 5m and 5p
    print(f"Opp2 riichi          : {vec[IDX_OPP2_RIICHI]}") # 1.0
    print(f"All finite           : {np.all(np.isfinite(vec))}")
import numpy as np
from mahjong.shanten import Shanten

NUM_TILE_TYPES = 34
MAX_TURNS = 18
MAX_SHANTEN = 8

AKA_5M, AKA_5P, AKA_5S = 16, 52, 88

# CHANGED: Increased to 240 dimensions to original map data + strategy + open melds
FEATURE_DIM = 240 

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

# --- STRATEGIC CONTEXT HELPERS ---

def encode_relative_points(record: dict) -> np.ndarray:
    """
    Encodes current points normalized, and the point differentials 
    relative to Shimoncha, Toimen, and Kamicha.
    """
    player_wind = record["player_wind"]
    my_pts = record.get(str(player_wind), {}).get("points", 25000)
    
    # Base scaling for self points (Centered around starting pool)
    encoded = [my_pts / 50000.0]
    
    # Calculate point distance relative to opponents in turn order sequence
    for offset in [1, 2, 3]:
        opp = (player_wind + offset) % 4
        opp_pts = record.get(str(opp), {}).get("points", 25000)
        diff = my_pts - opp_pts
        encoded.append(diff / 20000.0)
        
    return np.array(encoded, dtype=np.float32)

def encode_table_stakes(record: dict) -> np.ndarray:
    """Encodes Honba count and Riichi sticks currently sitting on the table."""
    honba_scaled = min(record.get("num_honba", 0) / 5.0, 1.0)
    riichi_sticks_scaled = min(record.get("num_riichi", 0) / 3.0, 1.0)
    return np.array([honba_scaled, riichi_sticks_scaled], dtype=np.float32)

def check_is_dealer(record: dict) -> float:
    """Returns 1.0 if the player is currently the dealer (Oya), else 0.0."""
    player_wind = record["player_wind"]
    dealer_wind = record.get("round_wind", 0) % 4
    return 1.0 if player_wind == dealer_wind else 0.0

# --- OPEN STATE / OWN MELDS HELPERS ---

def encode_own_melds(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Tracks which tiles are sitting open in your own melds block,
    and returns a one-hot count of how many open calls you've made.
    """
    melds_34 = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    player_wind = str(record["player_wind"])
    
    my_public_info = record.get(player_wind, {})
    melds_list = my_public_info.get("melds", [])
    
    meld_count = min(len(melds_list), 4)
    count_one_hot = np.zeros(4, dtype=np.float32)
    if meld_count > 0:
        count_one_hot[meld_count - 1] = 1.0
        
    for meld in melds_list:
        for tile in meld.get("tiles", []):
            if tile >= 0:
                melds_34[tile_to_34(tile)] += 1.0
                
    return melds_34, count_one_hot

# --- MAIN COMPREHENSIVE ENCODER ---

def encode_state_vector(record: dict) -> np.ndarray:
    """
    Encodes the comprehensive board state into a 240-dimensional vector.
    
    Layout:
      [0:187]   -> Original features (Hands, Doras, Winds, Opponent Discards/Riichi)
      [187:202] -> Positional Context (Point diffs, Table values, Dealer status)
      [202:240] -> Self Melds Context (Open tiles tracking, open meld counters)
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    hand = record["hand_tiles"]

    # =========================================================================
    # [0:187] Original Feature Block
    # =========================================================================
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

    # =========================================================================
    # [187:202] Strategic Context Block
    # =========================================================================
    # 187, 188, 189, 190: Scoring configuration
    vec[187:191] = encode_relative_points(record)
    
    # 191, 192: Table values
    vec[191:193] = encode_table_stakes(record)
    
    # 193: Dealer status flag
    vec[193] = check_is_dealer(record)
    
    # 194-197: One-hot absolute round wind sub-index (E1-E4 match 0-3, etc.)
    vec[194:198] = encode_wind(record.get("round_wind", 0) % 4)
    
    # 198: True linear wall depth percentage
    vec[198] = remain_tiles / 136.0
    
    # Note: Indices 199, 200, 201 are reserved zero paddings ensuring safety buffer

    # =========================================================================
    # [202:240] Open Hand / Self Melds Block
    # =========================================================================
    own_melds_array, melds_count_one_hot = encode_own_melds(record)
    
    # 202 to 235: Matrix mapping tiles sitting in your open melds
    vec[202:236] = own_melds_array
    
    # 236 to 239: One-hot indicating number of open melds made (1 to 4)
    vec[236:240] = melds_count_one_hot

    return vec
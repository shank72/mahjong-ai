import numpy as np

def state_to_features(state):
    """
    Convert MahjongState into numeric feature vector of length 375.
    """

    features = []
    
    # Your hand (34)
    features.extend(state.hand)                 
    
    # Visible tiles on table (34)
    visible = state.visible_tiles()             
    features.extend(visible)

    # Remaining unseen tiles (34)
    features.extend(state.remaining_tiles())    

    # Discards per player (4 x 34)
    for p in range(4):
        features.extend(state.discards[p])

    # Melds per player (4 x 34)
    for p in range(4):
        features.extend(state.melds[p])

    # Shanten number (1)
    features.append(state.shanten())

    # print(len(features))

    return np.array(features, dtype=np.float32)
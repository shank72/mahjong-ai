import copy

class HeuristicAgent:
    def choose_discard(self, state):
        best_tile = None
        best_shanten = 999

        for tile in range(34):
            if state.hand[tile] > 0:
                new_hand = copy.deepcopy(state.hand)
                new_hand[tile] -= 1

                temp_state = copy.deepcopy(state)
                temp_state.hand = new_hand

                s = temp_state.shanten()

                if s < best_shanten:
                    best_shanten = s
                    best_tile = tile

        return best_tile
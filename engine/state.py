from mahjong.shanten import Shanten

class MahjongState:
    """
    Represents everything the agent knows at a moment in the game.
    All tile data is represented as length-34 count vectors, 
    since there are 34 distinct tile types in Mahjong.
    """

    def __init__(self, hand_34, discards_4x34, melds_4x34):
        """
        hand_34:           [34]     your hand tile counts
        discards_4x34:     [4][34]  discard history per player
        melds_4x34:        [4][34]  exposed meld tiles per player (chi/pon/kan)
        """
        self.hand = hand_34
        self.discards = discards_4x34
        self.melds = melds_4x34

        self.shanten_calculator = Shanten()
        self._validate_shapes()

    # Validation
    def _validate_shapes(self):
        assert len(self.hand) == 34

        assert len(self.discards) == 4
        for p in range(4):
            assert len(self.discards[p]) == 34

        assert len(self.melds) == 4
        for p in range(4):
            assert len(self.melds[p]) == 34

    # Derived Information
    def visible_tiles(self):
        """
        Tiles visible on the table from discards and melds.
        """
        visible = [0] * 34

        for p in range(4):
            for i in range(34):
                visible[i] += self.discards[p][i]
                visible[i] += self.melds[p][i]

        return visible

    def remaining_tiles(self):
        """
        Estimate how many of each tile remain unseen.
        4 copies of each tile exist in Mahjong.
        """
        visible = self.visible_tiles()

        remaining = []
        for i in range(34):
            remaining.append(4 - visible[i] - self.hand[i])

        return remaining

    def shanten(self):
        return self.shanten_calculator.calculate_shanten(self.hand)
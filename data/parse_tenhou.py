"""
data/parse_tenhou.py

Parses raw Tenhou .mjlog XML files into a list of game state dicts,
one dict per discard decision made during a game.

Each output record looks like:
{
    "hand":              [int, ...],   # 14 tile IDs before discard (0-135, Tenhou encoding)
    "discard":           int,          # tile ID the player discarded (the label)
    "dora_indicators":   [int, ...],   # list of dora indicator tile IDs
    "seat_wind":         int,          # 0=East 1=South 2=West 3=North
    "round_wind":        int,          # 0=East 1=South
    "turn":              int,          # 0-indexed draw number this round
    "opponent_discards": [[int], [int], [int]],  # each opponent's visible discards so far
    "opponent_riichi":   [bool, bool, bool],     # whether each opponent has declared riichi
    "player_seat":       int,          # absolute seat of the acting player (0-3)
}

Usage:
    python data/parse_tenhou.py --in data/raw/ --out data/parsed/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TENHOU TILE ENCODING  (136 tiles total; each type has 4 copies)
Each tile type occupies a block of 4 consecutive IDs.
tile_type = tile_id // 4   →   collapses 136 IDs → 34 types.

MAN (characters)  tile_id 0-35      tile_type 0-8
  1m:  0- 3    2m:  4- 7    3m:  8-11
  4m: 12-15    5m: 16-19*   6m: 20-23
  7m: 24-27    8m: 28-31    9m: 32-35
  * tile_id 16 is the aka dora (red 5-man)

PIN (circles)     tile_id 36-71     tile_type 9-17
  1p: 36-39    2p: 40-43    3p: 44-47
  4p: 48-51    5p: 52-55*   6p: 56-59
  7p: 60-63    8p: 64-67    9p: 68-71
  * tile_id 52 is the aka dora (red 5-pin)

SOU (bamboo)      tile_id 72-107    tile_type 18-26
  1s: 72-75    2s: 76-79    3s: 80-83
  4s: 84-87    5s: 88-91*   6s: 92-95
  7s: 96-99    8s:100-103   9s:104-107
  * tile_id 88 is the aka dora (red 5-sou)

HONORS            tile_id 108-135   tile_type 27-33
  East:  108-111  (tile_type 27)
  South: 112-115  (tile_type 28)
  West:  116-119  (tile_type 29)
  North: 120-123  (tile_type 30)
  Haku:  124-127  (tile_type 31)   white dragon
  Hatsu: 128-131  (tile_type 32)   green dragon
  Chun:  132-135  (tile_type 33)   red dragon

DRAW / DISCARD TAG LETTERS (encoded as the XML tag name itself):
  Draw tags:    T=seat0  U=seat1  V=seat2  W=seat3
  Discard tags: D=seat0  E=seat1  F=seat2  G=seat3
  Example: <T16/> = seat 0 drew tile 16 (aka 5-man)
           <D16/> = seat 0 discarded tile 16
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import os
import gzip
from pathlib import Path
from lxml import etree
from tqdm import tqdm


# ── tile helpers ─────────────────────────────────────────────────────────────

def tile_to_34(tile136: int) -> int:
    """Convert Tenhou 136-tile encoding to 34-tile type index."""
    return tile136 // 4


def normalize_tile(tile136: int) -> int:
    """
    Normalize aka dora (red fives) to their standard equivalents.
    16 -> 16 (already 5m), 52 -> 52 (5p), 88 -> 88 (5s)
    Tenhou uses specific IDs for red fives; we keep them as-is
    since tile_to_34 handles them correctly (16//4=4 → 5m index).
    """
    return tile136


def hand_from_init(init_str: str) -> list[int]:
    """Parse the initial 13-tile hand from an <INIT> hai attribute string."""
    return [int(t) for t in init_str.split(",") if t]


def parse_draw_tag(tag: str) -> tuple[int, int] | None:
    """
    Tenhou encodes draws as single-letter tags: T=seat0, U=seat1, V=seat2, W=seat3
    followed by the tile ID. Returns (seat, tile_id) or None if not a draw tag.

    IMPORTANT — information asymmetry:
    In a real game you only see YOUR OWN drawn tile; opponent draws are hidden.
    We parse all seats here to keep hand state accurate internally, but
    feature_encoder.py must NEVER include opponent tile IDs in the feature vector.
    Opponent draw counts are inferred from turn number only — not tile identity.
    """
    if len(tag) < 2:
        return None
    if tag[0] in "TUVW" and tag[1:].isdigit():
        seat = "TUVW".index(tag[0])
        return seat, int(tag[1:])
    return None


def parse_discard_tag(tag: str) -> tuple[int, int] | None:
    """
    Tenhou encodes discards as: D=seat0, E=seat1, F=seat2, G=seat3
    followed by tile ID. Returns (seat, tile_id) or None.
    """
    if len(tag) < 2:
        return None
    if tag[0] in "DEFG" and tag[1:].isdigit():
        seat = "DEFG".index(tag[0])
        return seat, int(tag[1:])
    return None


# ── round parser ─────────────────────────────────────────────────────────────

def parse_round(round_elem) -> list[dict]:
    """
    Parse one <ROUND> (kyoku) element and return a list of decision records,
    one per discard made by any player.
    """
    records = []

    # -- round metadata from <INIT> --
    init = round_elem.find("INIT")
    if init is None:
        return records

    seed = init.get("seed", "0,0,0,0,0,0").split(",")
    round_wind = int(seed[0]) // 4          # 0=East, 1=South
    dealer_seat = int(seed[0]) % 4

    # dora indicators (can grow during the round on kans)
    initial_doras = [int(seed[5])]
    dora_indicators = list(initial_doras)

    # starting hands for each seat
    hands: list[list[int]] = [[], [], [], []]
    for seat in range(4):
        hai_str = init.get(f"hai{seat}", "")
        if hai_str:
            hands[seat] = hand_from_init(hai_str)

    # seat wind = (seat - dealer) % 4
    seat_winds = [(s - dealer_seat) % 4 for s in range(4)]

    # opponent discard tracking
    opponent_discards: list[list[int]] = [[], [], [], []]
    riichi_flags: list[bool] = [False, False, False, False]

    # current drawn tile per seat (set on draw, cleared on discard)
    drawn: dict[int, int] = {}

    # turn counter per seat
    turn_count: list[int] = [0, 0, 0, 0]

    # -- iterate child elements in document order --
    for elem in round_elem:
        tag = elem.tag

        # draw
        result = parse_draw_tag(tag)
        if result is not None:
            seat, tile = result
            drawn[seat] = tile
            hands[seat].append(tile)
            turn_count[seat] += 1
            continue

        # discard
        result = parse_discard_tag(tag)
        if result is not None:
            seat, tile = result

            # build opponents lists (relative seats 1,2,3 away)
            opp_seats = [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]
            opp_discards = [opponent_discards[s] for s in opp_seats]
            opp_riichi = [riichi_flags[s] for s in opp_seats]

            # record the decision (hand still has 14 tiles including drawn)
            hand_snapshot = [t for t in hands[seat] if t != tile]  # 13 tiles after discard

            records.append({
                "hand":              hands[seat][:],   # 14 tiles before discard
                "discard":           tile,
                "dora_indicators":   dora_indicators[:],
                "seat_wind":         seat_winds[seat],
                "round_wind":        round_wind,
                "turn":              turn_count[seat] - 1,
                "opponent_discards": [d[:] for d in opp_discards],
                "opponent_riichi":   opp_riichi[:],
                "player_seat":       seat,
            })

            # update state
            if tile in hands[seat]:
                hands[seat].remove(tile)
            opponent_discards[seat].append(tile)
            drawn.pop(seat, None)
            continue

        # riichi declaration
        if tag == "REACH":
            seat = int(elem.get("who", -1))
            step = int(elem.get("step", 0))
            if step == 2:   # step=2 means riichi is accepted
                riichi_flags[seat] = True
            continue

        # new dora revealed (after kan)
        if tag == "DORA":
            hai = elem.get("hai")
            if hai is not None:
                dora_indicators.append(int(hai))
            continue

    return records


# ── file parser ──────────────────────────────────────────────────────────────

def parse_mjlog(path: Path) -> list[dict]:
    """
    Parse a single .mjlog or .mjlog.gz file.
    Returns a flat list of decision records across all rounds.
    """
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                content = f.read()
        else:
            content = path.read_bytes()

        root = etree.fromstring(content)
    except Exception as e:
        print(f"[warn] failed to parse {path.name}: {e}")
        return []

    records = []
    for round_elem in root.findall("ROUND"):
        records.extend(parse_round(round_elem))
    return records


# ── batch runner ─────────────────────────────────────────────────────────────

def parse_directory(in_dir: Path, out_dir: Path) -> None:
    """
    Parse all .mjlog / .mjlog.gz files in in_dir.
    Writes one .jsonl file per source file into out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(
        list(in_dir.glob("**/*.mjlog")) +
        list(in_dir.glob("**/*.mjlog.gz"))
    )

    if not log_files:
        print(f"No .mjlog files found in {in_dir}")
        return

    total_records = 0
    for log_path in tqdm(log_files, desc="Parsing logs"):
        records = parse_mjlog(log_path)
        if not records:
            continue

        out_path = out_dir / (log_path.stem + ".jsonl")
        with open(out_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        total_records += len(records)

    print(f"\nDone. {total_records:,} decision records from {len(log_files)} files.")
    print(f"Output: {out_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Tenhou mjlog files into decision records.")
    parser.add_argument("--in",  dest="in_dir",  required=True, help="Directory of raw .mjlog files")
    parser.add_argument("--out", dest="out_dir", required=True, help="Directory to write .jsonl output")
    args = parser.parse_args()

    parse_directory(Path(args.in_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
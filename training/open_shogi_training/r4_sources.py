"""Pinned, bounded R4 public data acquisition and PackedSfenValue root decoding.

Format reference: YaneuraOu source/extra/sfen_packer.cpp (AGPL-3.0).
Dataset terms are audited separately in docs/source-audits/r4-c2.md.
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from .evaluator_data import digest, symmetry_keys
from .phase10r_model import parse_sfen

PSV = struct.Struct("<32shHHbB")
BOARD_CODES = {
    (0, 1): "",
    (1, 2): "P",
    (3, 4): "L",
    (11, 4): "N",
    (7, 4): "S",
    (15, 5): "G",
    (31, 6): "B",
    (63, 6): "R",
}
HAND_CODES = {(code >> 1, bits - 1): piece for (code, bits), piece in BOARD_CODES.items() if piece}


def unpack_position(raw: bytes, ply: int = 1) -> str:
    if len(raw) != 32 or not 0 <= ply < 65536:
        raise ValueError("invalid PackedSfen length/ply")
    value, cursor = int.from_bytes(raw, "little"), 0

    def take(bits):
        nonlocal cursor
        if cursor + bits > 256:
            raise ValueError("truncated PackedSfen")
        result = (value >> cursor) & ((1 << bits) - 1)
        cursor += bits
        return result

    def piece(hand=False):
        code = 0
        table = HAND_CODES if hand else BOARD_CODES
        for bits in range(1, 7):
            code |= take(1) << (bits - 1)
            if (code, bits) in table:
                return table[code, bits]
        raise ValueError("invalid PackedSfen piece code")

    turn = take(1)
    kings = [take(7), take(7)]
    if max(kings) >= 81 or kings[0] == kings[1]:
        raise ValueError("PackedSfen requires two distinct kings")
    board = [""] * 81
    for side, square in enumerate(kings):
        board[square] = "k" if side else "K"
    for square in range(81):
        if square in kings:
            continue
        name = piece()
        if name:
            promoted = take(1) if name != "G" else 0
            side = take(1)
            board[square] = ("+" if promoted else "") + (name.lower() if side else name)
    hands = [Counter(), Counter()]
    while cursor < 256:
        name = piece(True)
        promoted = take(1) if name != "G" else 0
        side = take(1)
        if promoted:
            # Missing-piece encodings are not part of this reviewed even-game corpus.
            raise ValueError("handicap PackedSfen is outside the admitted source")
        hands[side][name] += 1
    ranks = []
    for rank in range(9):
        text, empty = "", 0
        for file in range(8, -1, -1):
            name = board[file * 9 + rank]
            if name:
                text += (str(empty) if empty else "") + name
                empty = 0
            else:
                empty += 1
        ranks.append(text + (str(empty) if empty else ""))
    hand = "".join(
        (str(hands[side][name]) if hands[side][name] > 1 else "") + (name.lower() if side else name)
        for side in (0, 1)
        for name in "RBGSNLP"
        if hands[side][name]
    )
    sfen = f"{'/'.join(ranks)} {'w' if turn else 'b'} {hand or '-'} {max(1, ply)}"
    parsed = parse_sfen(sfen)
    pieces = Counter(
        p.kind if p.kind < 8 else {8: 0, 9: 1, 10: 2, 11: 3, 12: 5, 13: 6}[p.kind]
        for p in parsed.board
        if p
    )
    for side in parsed.hands:
        pieces.update({i: n for i, n in enumerate(side)})
    if [pieces[i] for i in range(8)] != [18, 4, 4, 4, 4, 2, 2, 2]:
        raise ValueError("PackedSfen material count mismatch")
    return sfen


def packed_move(move: int) -> str:
    destination, origin = move & 127, (move >> 7) & 127

    def square(s):
        return f"{s // 9 + 1}{chr(97 + s % 9)}"

    if destination >= 81:
        raise ValueError("invalid packed move")
    if move & 16384:
        if not 1 <= origin <= 7 or move & 32768:
            raise ValueError("invalid packed drop")
        return "PLNSBRG"[origin - 1] + "*" + square(destination)
    if origin >= 81 or origin == destination:
        raise ValueError("invalid packed board move")
    return square(origin) + square(destination) + ("+" if move & 32768 else "")


def acquire(root: Path, plan: dict) -> list[Path]:
    from .r4_data import reference

    paths = []
    for shard in plan["shards"]:
        path = root / shard["path"]
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(
            root.resolve() / "local"
        ):
            raise ValueError("unsafe source output")
        if path.exists():
            reference(root, path, shard["sha256"])
            paths.append(path)
            continue
        url = shard["url"]
        if not url.startswith("https://huggingface.co/datasets/nodchip/shogi_hao_depth9/resolve/"):
            raise ValueError("source URL outside the reviewed dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        if part.is_symlink():
            raise ValueError("linked source staging file")
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as f:
                    # urlopen follows redirects; the payload hash still pins the
                    # bytes, but the final hop must stay on the reviewed dataset
                    # hosts (huggingface.co and its LFS/Xet CDN domains).
                    final = urllib.parse.urlsplit(response.geturl())
                    host = final.hostname or ""
                    if final.scheme != "https" or not (
                        host == "huggingface.co"
                        or host.endswith(".huggingface.co")
                        or host.endswith(".hf.co")
                    ):
                        raise ValueError("source redirect left the reviewed dataset hosts")
                    if response.status != 200:
                        raise ValueError("full pinned shard required")
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > shard["bytes"]:
                            raise ValueError("source exceeds declared size")
                        f.write(block)
                if total != shard["bytes"] or digest(part) != shard["sha256"]:
                    raise ValueError("source size/hash mismatch")
                part.rename(path)
                break
            except (OSError, ValueError):
                if attempt == 1:
                    raise
        paths.append(path)
        print(json.dumps({"acquired": shard["path"], "bytes": total}), flush=True)
    return paths


def rows(path: Path, shard: dict, stride: int):
    """Infer contiguous segments from unshuffled gamePly resets; keep uncertainty explicit."""
    if path.stat().st_size % PSV.size or stride < 2:
        raise ValueError("invalid PSV source or spacing")
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        previous, start, family, split, last_selected = 65536, 0, "", "", -stride
        for index in range(len(data) // PSV.size):
            raw, score, movement, ply, result, padding = PSV.unpack_from(data, index * PSV.size)
            if result not in (-1, 0, 1) or not 1 <= ply <= 512:
                raise ValueError(f"invalid PSV metadata at row {index}")
            if ply <= previous:
                start, last_selected = index, -stride
                initial = unpack_position(raw, ply)
                family = min(symmetry_keys(initial))
                bucket = int(family[:8], 16) % 100
                split = (
                    "validation" if bucket < 3 else "development_test" if bucket < 6 else "train"
                )
            previous = ply
            if ply - last_selected < stride or abs(score) >= 26099:
                continue
            # Derive an offset from the segment identity, avoiding every game's first row.
            if ply < 8 + int(family[8:10], 16) % stride:
                continue
            last_selected = ply
            sfen = unpack_position(raw, ply)
            yield {
                "sfen": sfen,
                "score": {"kind": "cp", "value": int(score * 100 / 90)},
                "raw_teacher_value": score,
                "scale": (
                    "YaneuraOu Value to USI cp: trunc(100*value/90); not probability calibration"
                ),
                "score_perspective": "side_to_move",
                "kind": "root",
                "move16": movement,
                "move": packed_move(movement),
                "game": f"hao:{shard['sha256']}:{start}",
                "family": f"hao:{family}",
                "group": "opening" if ply <= 40 else "general" if ply <= 100 else "attack_end",
                "ply": ply,
                "split": split,
                "source": "nodchip_hao_depth9",
                "source_sha256": shard["sha256"],
                "source_row": index,
                "teacher_label_origin": "public_hao_depth9_root_pv_leaf_value",
                "round_source": 1,
                "multipv": None,
                "usi_bound": "not_encoded",
                "mate_signal": "excluded",
                "history_context": (
                    "not_encoded; inferred contiguous segment, not certified game ID"
                ),
                "factual_result_unused": result,
                "padding_unused": padding,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--acquire-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text())
    plan = config["data_preparation"]
    acquire(root, plan)
    if not args.acquire_only:
        from .r4_data import build_pool

        print(json.dumps(build_pool(root, plan, root / plan["output"]), indent=2))


if __name__ == "__main__":
    main()

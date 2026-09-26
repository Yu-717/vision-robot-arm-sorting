"""Put the blocks back out on the map at random, so a statistics session does not need a pair of hands
between rounds.

    python3 src/tasks/scramble.py                  # scan, then move every block to a free marker
    python3 src/tasks/scramble.py --dry-run        # choose the markers and print them, no motion
    python3 src/tasks/scramble.py --seed 7         # repeatable draw

Targets are marker centres, never the bin row: a marker centre is a position known to a millimetre from
the printed map, so how far a block lands from it measures the place, and starting a block on its own bin
would leave nothing to sort. Two blocks are kept at least 8.5 cm apart -- the grid is 6 x 5 cm, and a 3 cm
block plus a finger either side is about 6 cm wide, so neighbouring markers would put a finger into the
block next door.

What this cannot do: turn a block over. The arm drops a tilted block (2026-09-25, twice), so every round
of an unattended session sees the same face of each block. Turning them stays a job for the operator.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("pipeline", "tasks", "tools", "spatial", "vision"):
    sys.path.insert(0, str(ROOT / "src" / sub))

import console_grasp as G  # noqa: E402
import sort_blocks as SB  # noqa: E402

MIN_APART_CM = 8.5          # 6 x 5 cm grid: this rules out the neighbouring markers, not the next ones out
DRAW_TRIES = 200


def bin_markers() -> set[str]:
    return set(SB.SORTING["category_bins"].values())


def candidates() -> list[str]:
    """Marker centres a block can be put on and picked up again: off the bin row, and with a reachable
    grasp. Yaw is unknown until the block is down, so reachability is checked without roll alignment."""
    out = []
    for marker in sorted(G.MAP["layout_m"], key=lambda k: (len(k), k)):
        if not marker.isdigit() or marker in bin_markers():
            continue
        mx, my = G.marker_centre(marker)
        bx, by = G.map_to_base(mx, my)
        if G.plan_approach(bx, by, None) is not None:
            out.append(marker)
    return out


def draw(n: int, rng: random.Random, pool: list[str] | None = None) -> list[str]:
    """n markers at least MIN_APART_CM apart. Raises when the pool cannot seat them, rather than
    quietly returning a tighter layout that the fingers would knock over."""
    pool = candidates() if pool is None else pool
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} reachable markers for {n} blocks")
    centres = {m: [v * 100 for v in G.marker_centre(m)] for m in pool}
    for _ in range(DRAW_TRIES):
        picked: list[str] = []
        for marker in rng.sample(pool, len(pool)):
            if all(math.dist(centres[marker], centres[p]) >= MIN_APART_CM for p in picked):
                picked.append(marker)
            if len(picked) == n:
                return picked
    raise ValueError(f"no layout of {n} markers {MIN_APART_CM} cm apart in {len(pool)} candidates")


def move(cube: dict, marker: str, plan_for, session) -> dict:
    """One block to one marker centre, through the verified stages. The caller keeps the identity: the
    block that was at `cube` is the block now on `marker`, which is what makes a round self-labelling."""
    plan = plan_for(cube)
    if plan is None:
        raise G.ConsoleError(f"no reachable grasp at {cube['map_cm']}")
    G.PLAN.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    started = time.time()
    G.hover(session)
    G.grasp(session)
    G.put(marker, session)
    target = [round(v * 100, 1) for v in G.marker_centre(marker)]
    G.log("scramble_moved", frm=cube["map_cm"], marker=marker, target_map_cm=target,
          category=cube.get("category"), seconds=round(time.time() - started))
    return dict(marker=marker, target_map_cm=target, frm=cube["map_cm"], category=cube.get("category"))


def scramble(cubes: list[dict], plan_for, session, rng: random.Random) -> list[dict]:
    """Every block to a free marker. Blocks are moved farthest-first: a block already standing on a marker
    that another block has drawn would be in the way, and moving the far ones first keeps the arm from
    reaching over what it just put down."""
    markers = draw(len(cubes), rng)
    order = sorted(range(len(cubes)), key=lambda i: -math.dist(cubes[i]["map_cm"], [0, 0]))
    moved = []
    for i, marker in zip(order, markers):
        moved.append(move(cubes[i], marker, plan_for, session))
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="draw and print the markers, move nothing")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scan", type=Path, help="use this scan instead of scanning now")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.dry_run and not args.scan:
        pool = candidates()
        print(f"{len(pool)} 个可用地图码: {' '.join(pool)}")
        print("抽到:", " ".join(draw(4, rng, pool)))
        return

    import field_scan as SC
    import run as R
    with G.Session() as sess:
        scan = args.scan.resolve() if args.scan else SC.scan(session=sess)
        scene = R.build_scene(scan)
        R.use_registration(scene["registration"])
        if args.dry_run:
            print("抽到:", " ".join(draw(len(scene["cubes"]), rng)))
            return
        for m in scramble(scene["cubes"], R.plan_for, sess, rng):
            print(f"{str(m['category'] or '?'):11s} {m['frm']} -> 码 {m['marker']} {m['target_map_cm']}")
        G.observe(sess)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the real sorting pipeline (src/pipeline/run.py) against the simulation back end.

    uv run --with 'opencv-python<5' --with numpy --with onnxruntime src/sim/run_pipeline.py --rounds 2

Start src/sim/sim_server.py first (map_a4_v2, no bins: the pipeline still puts blocks on map codes
40/41/43/44). Nothing in the pipeline is changed: the same scan, scene building, planning, grasping and
logging code runs, only pointed at the simulation's address, and its files go under data/sim/ instead of
data/runs/, so simulated events never mix with the real arm's log.

Each round puts fresh blocks at random, scans, sorts, and writes the same events the real arm writes to
data/sim/log.jsonl -- the file the simulation serves at /api/log.
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
SIM = ROOT / "data" / "sim"
CATEGORIES = ("recyclable", "kitchen", "hazardous", "other")
# Where a round's blocks are put: the band the real rounds use (map rows 8-18 cm). Nearer than about
# y = 8 cm only the folded near views see a block, the known weak strip of the scan (docs/10, risk 1).
X_CM, Y_CM, APART_CM, CLEAR_OF_BIN_CM = (-11.0, 11.0), (9.0, 17.0), 5.0, 3.2


def lay_out(count: int, rng: random.Random, bins_cm: list[list[float]]) -> list[dict]:
    spots: list[tuple[float, float]] = []
    for _ in range(count):
        for _ in range(500):
            p = (rng.uniform(*X_CM), rng.uniform(*Y_CM))
            if all(math.dist(p, q) >= APART_CM for q in spots) and all(math.dist(p, b) >= CLEAR_OF_BIN_CM for b in bins_cm):
                spots.append(p)
                break
        else:
            raise SystemExit("no room for that many blocks")
    return [dict(category=CATEGORIES[i % 4], map_cm=[round(x, 1), round(y, 1)], yaw_deg=round(rng.uniform(-45, 45)),
                 up=rng.randrange(5)) for i, (x, y) in enumerate(spots)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8770", help="the simulation back end")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, help="for repeatable block layouts")
    parser.add_argument("--grip-miss", type=float, help="set the simulation's share of grips that close on nothing")
    parser.add_argument("--keep-scene", action="store_true", help="sort what is on the table instead of resetting it")
    parser.add_argument("--dry-run", action="store_true", help="scan and plan only")
    args = parser.parse_args()

    import console_step
    console_step.BASE = args.base.rstrip("/")           # before the modules below copy it
    import console_grasp as G
    import field_scan as SC
    import run as R

    boot = console_step.call("/api/bootstrap")
    if boot.get("status", {}).get("mode") != "simulation" or "sim" not in boot["status"]:
        raise SystemExit(f"{args.base} is not the simulation back end; refusing to run")
    G.RUNS = SIM
    G.PLAN = SIM / "plan.json"
    SC.RUNS = SIM / "pipeline"
    SIM.mkdir(parents=True, exist_ok=True)

    if args.grip_miss is not None:
        fault = console_step.call("/api/sim/faults", dict(grip_miss=args.grip_miss), token=boot["token"])
        if fault.get("error"):
            raise SystemExit(fault["error"])
    import sort_blocks as SB
    bins_cm = [[v * 100 for v in G.marker_centre(m)] for m in SB.SORTING["category_bins"].values()]
    rng = random.Random(args.seed)
    for i in range(args.rounds):
        if not args.keep_scene:
            for request in [("/api/sim/reset", dict(count=0))] + [("/api/sim/block", b) for b in lay_out(args.blocks, rng, bins_cm)]:
                world = console_step.call(*request, token=boot["token"])
                if world.get("error"):
                    raise SystemExit(world["error"])
        G.require_empty_gripper()
        with G.Session() as sess:
            run = SC.scan(session=sess)
            scene = R.build_scene(run)
            R.use_registration(scene["registration"])
            if not args.dry_run:
                scene = R.refine(run, scene, sess)
                R.use_registration(scene["registration"])
            jobs = R.decide(scene)
            counts = None if args.dry_run else R.execute(run, scene, jobs, sess)
        truth = console_step.call("/api/sim/world")
        (run / "world_after.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False))
        print(f"round {i + 1}/{args.rounds}: {counts}  ->  {run.relative_to(ROOT)}")
        time.sleep(1)


if __name__ == "__main__":
    main()

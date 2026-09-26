"""Camera-only sorting pipeline: scan -> register the map -> find cubes -> classify -> grasp and place.

    python3 src/pipeline/run.py                      # scan now, plan, sort (moves the arm; operator watching)
    python3 src/pipeline/run.py --scan <run_dir>     # reuse a scan
    python3 src/pipeline/run.py --dry-run            # scan + plan only; writes scene.json and scene.png

The map is registered from this scan (no touching, no phone). Cube centre and yaw come from the top-plane
orthophoto (src/pipeline/cubes.py); category from the official 3-class model (its detections from the scan
views, projected onto the top plane) when confident and consistent with the top colour, else the colour
(src/tasks/sort_blocks.resolve). The motion reuses the verified console stages (hover, grasp, put) with
the plan made from the scan instead of a close look.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("pipeline", "tasks", "tools", "spatial", "vision"):
    sys.path.insert(0, str(ROOT / "src" / sub))

import console_grasp as G  # noqa: E402
import cubes as C  # noqa: E402
import scene as S  # noqa: E402
import sort_blocks as SB  # noqa: E402
from block_detector import hue_label  # noqa: E402

COLOURS = ("red", "green", "blue", "yellow")
DET_MERGE_CM = 2.5       # detections of one block from different views land within this on the mid plane
DET_MIN_CONF = 0.4
DET_STRONG_SHARE = 0.6   # the leading category's share of the summed confidences
DET_STRONG_SUM = 0.8
DET_CUT_MERGE_CM = 3.5   # a box cut by the image border casts its centre off the block's
DET_SAME_BLOCK_CM = 3.0  # a cluster this near an outline-found block is that block
DET_SINGLE_VIEW_CONF = 0.75
ORTHO_AGREE_CM = 1.2     # else the orthophoto square latched onto a side face (blue, 2.1 cm off, 2026-09-25)
MIN_SEPARATION_CM = 3.2  # two 3 cm blocks cannot be nearer; a nearer pair is one block smeared on the plane
SMEAR_CM = 5.0           # an outline with no detection this near a detected block is that block's own side


def top_colour(top: np.ndarray, cube: dict) -> tuple[str | None, float]:
    """Dominant saturated hue inside the inner 60 % of the top face, or None for a white/sticker top."""
    mask = np.zeros(top.shape[:2], np.uint8)
    k = C.square_kernel(cube["yaw_deg"], C.SIDE_PX * 0.6)
    h = k.shape[0] // 2
    r0, c0 = cube["row"] - h, cube["col"] - h
    if r0 < 0 or c0 < 0 or r0 + k.shape[0] > mask.shape[0] or c0 + k.shape[1] > mask.shape[1]:
        return None, 0.0
    mask[r0:r0 + k.shape[0], c0:c0 + k.shape[1]] = (k > 0).astype(np.uint8)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    inside = hsv[mask > 0]
    # V > 35: the green top reads V ~50 in the folded near views (2026-09-25); paper and shadows stay out on S
    saturated = inside[(inside[:, 1] > 90) & (inside[:, 2] > 35)]
    share = len(saturated) / max(1, len(inside))
    if share < 0.4:
        return None, round(share, 2)
    hues = saturated[:, 0].astype(float)
    if np.mean((hues < 12) | (hues > 165)) > 0.5:
        hues = np.where(hues < 90, hues + 180, hues)
    label = hue_label(float(np.median(hues)) % 180)
    return (label if label in COLOURS else None), round(share, 2)


def official_on_cubes(scan: S.Scan, cubes: list[dict]) -> None:
    """Attach to each cube the best official detection whose box centre, cast onto the top plane, lands
    within 2.5 cm of the cube centre (background detections land nowhere near a cube)."""
    for c in cubes:
        c["sticker"] = None
    for view, pose in zip(scan.views, scan.poses):
        for d in view.get("official") or []:
            if "bbox" not in d or d.get("conf", 0) < SB.SORTING["model_min_conf"]:
                continue
            x0, y0, x1, y1 = d["bbox"]
            p = pose.pixel_to_plane(((x0 + x1) / 2, (y0 + y1) / 2), scan.k, scan.dist, z=S.BLOCK)
            near = min(cubes, key=lambda c: math.dist(c["map_cm"], (p[0] * 100, p[1] * 100)), default=None)
            if near is None or math.dist(near["map_cm"], (p[0] * 100, p[1] * 100)) > 2.5:
                continue
            if near["sticker"] is None or d["conf"] > near["sticker"]["conf"]:
                near["sticker"] = dict(label=d["label"], category=d["category"], conf=d["conf"], view=view["image"])


def detector_clusters(scan: S.Scan) -> list[dict]:
    """The 4-class block detector (src/vision/blocks_detector.py) on every view; each box centre is cast
    onto the plane of the block centres (z = 1.5 cm), and the casts within DET_MERGE_CM are one block,
    with confidence votes per category. Boxes cut by the image border are left out: their centre is not
    the block's. [] when no detector can run here."""
    try:
        from blocks_detector import BlocksDetector
        det = BlocksDetector()
    except Exception as e:          # no model file, no onnxruntime / ais_bench
        print("block detector unavailable:", e)
        return []
    casts, cut = [], []
    for v, (img, pose) in enumerate(zip(scan.images, scan.poses)):
        h, w = img.shape[:2]
        for d in det.detect(img, conf=DET_MIN_CONF):
            x0, y0, x1, y1 = d["bbox"]
            p = pose.pixel_to_plane(((x0 + x1) / 2, (y0 + y1) / 2), scan.k, scan.dist, z=S.BLOCK / 2)
            c = dict(xy=[p[0] * 100, p[1] * 100], category=d["category"], conf=d["conf"], view=v)
            (cut if x0 <= 2 or y0 <= 2 or x1 >= w - 3 or y1 >= h - 3 else casts).append(c)
    clusters: list[dict] = []
    for c in sorted(casts, key=lambda c: -c["conf"]):
        near = [k for k in clusters if math.dist(k["xy"], c["xy"]) <= DET_MERGE_CM]
        if near:
            k = near[0]
            k["members"].append(c)
            wsum = sum(m["conf"] for m in k["members"])
            k["xy"] = [sum(m["xy"][i] * m["conf"] for m in k["members"]) / wsum for i in (0, 1)]
        else:
            clusters.append(dict(xy=list(c["xy"]), members=[c]))
    # A box cut by the image border has a biased centre, so it joins a whole box's cluster when there is
    # one nearby. Blocks near the map's edges are cut in every view, though (11 of 60 were lost this way,
    # 2026-09-26), so the rest cluster among themselves and are marked approximate: the orthophoto then
    # squares them up over a wider window.
    leftover = []
    for c in cut:
        near = min(clusters, key=lambda k: math.dist(k["xy"], c["xy"]), default=None)
        if near is not None and math.dist(near["xy"], c["xy"]) <= DET_CUT_MERGE_CM:
            near.setdefault("cut", []).append(c)
        else:
            leftover.append(c)
    for c in sorted(leftover, key=lambda c: -c["conf"]):
        near = [k for k in clusters if k.get("approx") and math.dist(k["xy"], c["xy"]) <= DET_CUT_MERGE_CM]
        if near:
            near[0]["members"].append(c)
        else:
            clusters.append(dict(xy=list(c["xy"]), members=[c], approx=True))
    for k in clusters:
        k["members"] += k.pop("cut", [])
    for k in clusters:
        votes: dict[str, float] = {}
        for m in k["members"]:
            votes[m["category"]] = votes.get(m["category"], 0.0) + m["conf"]
        k["votes"] = {c: round(v, 2) for c, v in sorted(votes.items(), key=lambda t: -t[1])}
        k["views"] = len({m["view"] for m in k["members"]})
        k["xy"] = [round(v, 2) for v in k["xy"]]
        k["approx"] = bool(k.get("approx"))
        del k["members"]
    return clusters


def categorize(cube: dict) -> tuple[str | None, str]:
    """A coloured top decides (it is the block's own colour); otherwise the detector's views vote; the
    official 3-class model only when neither says anything."""
    by_colour = SB.SORTING["colour_category"].get(cube["colour"]) if cube.get("colour") else None
    votes = (cube.get("detector") or {}).get("votes") or {}
    lead = max(votes, key=votes.get) if votes else None
    strong = bool(lead) and votes[lead] >= DET_STRONG_SUM and votes[lead] / sum(votes.values()) >= DET_STRONG_SHARE
    if by_colour:
        if strong and lead != by_colour:
            return by_colour, f"colour; detector disagreed ({votes})"
        return by_colour, "colour" + (", detector agrees" if strong else "")
    if strong:
        return lead, f"detector {votes} over {cube['detector']['views']} views"
    return SB.resolve(None, cube.get("sticker"))


def dedupe(cubes: list[dict]) -> list[dict]:
    """Drop a block too near a better-evidenced one: on the plane of the tops a block's own side smears
    downwards and is squared up as a second block 2.5-3.3 cm away (2026-09-26). Detector evidence outranks
    a bare outline, then the stronger outline."""
    order = sorted(cubes, key=lambda c: (c.get("detector") is not None, c["score"]), reverse=True)
    kept: list[dict] = []
    for c in order:
        # a block the detector never saw, next to one it did, is that block's own painted side cast onto
        # the top plane (a yellow side read as a second block 3.3 cm away, 2026-09-26)
        limit = SMEAR_CM if c.get("detector") is None else MIN_SEPARATION_CM
        if all(math.dist(c["map_cm"], k["map_cm"]) >= (limit if k.get("detector") is not None else MIN_SEPARATION_CM)
               for k in kept):
            kept.append(c)
    return [c for c in cubes if c in kept]


def build_scene(run: Path) -> dict:
    scan = S.Scan(run)
    reg = scan.register()
    scan.camera_poses()
    top, weight = scan.ortho(S.BLOCK)
    fg = scan.foreground(S.BLOCK)
    # the map difference only knows the paper: a cube half over the map's edge had fill 0.3 (green,
    # 2026-09-25 19:28). A saturated face is a cube wherever it lies (table S/V ~0/0, paper 2 % saturated).
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    coloured = ((hsv[..., 1] > 90) & (hsv[..., 2] > 35)).astype(np.float32)
    fg = np.maximum(fg, cv2.medianBlur((coloured * 255).astype(np.uint8), 5).astype(np.float32) / 255)
    found = C.find_cubes(top, weight, min_score=0.2, foreground=fg)
    cubes = []
    for c in found:
        x, y = S.Scan.to_map(c["col"], c["row"])
        yaw_map = (-(c["yaw_deg"]) + 45) % 90 - 45        # image rows run towards -y: the angle flips sign
        colour, share = top_colour(top, c)
        # coverage at the cube: a cube at the edge of what the views saw is only partly measured
        r0, r1, c0, c1 = max(0, c["row"] - 18), c["row"] + 18, max(0, c["col"] - 18), c["col"] + 18
        coverage = float((weight[r0:r1, c0:c1] > 0.05).mean())
        cubes.append(dict(map_cm=[round(x * 100, 2), round(y * 100, 2)], yaw_deg=round(yaw_map, 1), score=c["score"],
                          fill=c["fill"], colour=colour, colour_share=share, coverage=round(coverage, 2),
                          px=[c["col"], c["row"]], img_yaw=c["yaw_deg"]))
    # the detector: its evidence for the blocks found above, and the blocks the orthophoto missed (a white
    # sticker top on white paper has almost no outline: 1 of 4 found on 2026-09-25 22:10)
    # Match clusters to the blocks the outline search found, nearest pair first and one cluster per block:
    # a cluster whose centre is pulled off by clipped boxes would otherwise claim its neighbour's block, and
    # the neighbour's cluster would claim it back, losing a block entirely (blue at the map's edge, 2026-09-26).
    clusters = detector_clusters(scan)
    pairs = sorted(((math.dist(c["map_cm"], k["xy"]), ki, ci)
                    for ki, k in enumerate(clusters) for ci, c in enumerate(cubes)
                    if math.dist(c["map_cm"], k["xy"]) <= DET_SAME_BLOCK_CM), key=lambda p: p[0])
    taken_k: set[int] = set()
    taken_c: set[int] = set()
    matched: dict[int, tuple[dict, float]] = {}
    for dist, ki, ci in pairs:
        if ki in taken_k or ci in taken_c:
            continue
        taken_k.add(ki)
        taken_c.add(ci)
        matched[ki] = (cubes[ci], dist)     # the cube itself: the list is edited below
    for ki, k in enumerate(clusters):
        if ki in matched:
            cube, dist = matched[ki]
            cube["detector"] = k
            if dist <= ORTHO_AGREE_CM:
                continue
            cubes.remove(cube)              # far apart: re-square it where the detector puts it
        elif k["views"] < 2 and max(k["votes"].values()) < DET_SINGLE_VIEW_CONF:
            continue
        col, row = S.Scan.to_px(k["xy"][0] / 100, k["xy"][1] / 100)
        r = C.refine_at(top, weight, col, row, window_px=25 if k["approx"] else 12)
        found.append(dict(col=r["col"], row=r["row"], yaw_deg=r["yaw_deg"], score=r["score"], fill=None))
        x, y = S.Scan.to_map(r["col"], r["row"])
        colour, share = top_colour(top, r)
        r0, r1, c0, c1 = max(0, r["row"] - 18), r["row"] + 18, max(0, r["col"] - 18), r["col"] + 18
        cubes.append(dict(map_cm=[round(x * 100, 2), round(y * 100, 2)], yaw_deg=round((-r["yaw_deg"] + 45) % 90 - 45, 1),
                          score=r["score"], fill=None, colour=colour, colour_share=share,
                          coverage=round(float((weight[r0:r1, c0:c1] > 0.05).mean()), 2),
                          px=[r["col"], r["row"]], img_yaw=r["yaw_deg"], detector=k, found_by="detector"))
    cubes = dedupe(cubes)
    found = [f for f in found if any(c["px"] == [f["col"], f["row"]] for c in cubes)]
    official_on_cubes(scan, cubes)
    for c in cubes:
        c["category"], c["source"] = categorize(c)
    vis = C.draw(top, found)
    for i, c in enumerate(cubes):
        tag = f"{c['colour'] or '-'} {(c['category'] or '?')[:4]}"
        cv2.putText(vis, tag, (c["px"][0] - 18, c["px"][1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.imwrite(str(run / "scene.png"), cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    scene = dict(run=str(run.relative_to(ROOT) if run.is_relative_to(ROOT) else run), registration={k: v for k, v in reg.items() if k != "per_view"},
                 cubes=cubes, t=time.time())
    (run / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False))
    return scene


REFINE_BELOW_COVERAGE = 0.98
# 2026-09-26, 60 placed blocks: every block whose centre or yaw came out wrong had either a weak outline
# (score below 0.4) or was found only in boxes cut by the image border ("approx"), which are the blocks at
# the edge of what the scan saw and are smeared on the top-plane orthophoto. A close view fixes both.
REFINE_BELOW_SCORE = 0.4


def needs_refine(cube: dict) -> bool:
    return (cube["coverage"] < REFINE_BELOW_COVERAGE or cube["score"] < REFINE_BELOW_SCORE
            or bool((cube.get("detector") or {}).get("approx")))


def refine(run: Path, scene: dict, session) -> dict:
    """One extra view aimed at each cube that was only partly seen or weakly detected, then the scene
    is rebuilt from all views. A wrong yaw lands the fingers on the cube's corners, and a jammed grip
    once latched the gripper servo until a power cycle (docs/05, 2026-09-26)."""
    import field_scan as SC
    weak = [c for c in scene["cubes"] if needs_refine(c)]
    if not weak:
        return scene
    views = json.loads((run / "views.json").read_text())
    for c in weak:
        bx, by = G.map_to_base(c["map_cm"][0] / 100, c["map_cm"][1] / 100)
        look = G.look_pose(bx, by, G.status()["pose"])
        G.move_to(session, look, G.ROLL_ZERO)
        jpeg, s = SC.frame_at_rest()
        name = f"view_{len(views):02d}.jpg"
        (run / name).write_bytes(jpeg)
        board = s["localization"].get("board") or {}
        views.append(dict(image=name, readback=s["pose"], command=look + [G.ROLL_ZERO], markers=board.get("markers"),
                          official=board.get("official", []), t=time.time(), refine_for=c["map_cm"]))
        G.log("refine_view", cube=c["map_cm"], pose=s["pose"])
    (run / "views.json").write_text(json.dumps(views, indent=1, ensure_ascii=False))
    return build_scene(run)


def use_registration(reg: dict) -> None:
    """Map placement of this run for all base <-> map conversions and the table height."""
    G.DX, G.DY = reg["map_origin_base_cm"][0] / 100, reg["map_origin_base_cm"][1] / 100
    G.TABLE_Z = reg["map_origin_base_cm"][2] / 100


def plan_for(cube: dict, prefer: int = 150) -> dict | None:
    """The console_grasp plan for one cube, from the scan (no close look)."""
    bx, by = G.map_to_base(cube["map_cm"][0] / 100, cube["map_cm"][1] / 100)
    approach = G.plan_approach(bx, by, cube["yaw_deg"], prefer)
    if approach is None:
        return None
    return dict(colour=cube.get("colour"), created=time.time(), block_base_cm=[round(bx * 100, 2), round(by * 100, 2)],
                board=dict(map_cm=cube["map_cm"], sticker=cube.get("sticker")), block_yaw_deg=cube["yaw_deg"],
                block_centre_z_cm=round((G.TABLE_Z + G.BLOCK / 2) * 100, 2), **approach)


def decide(scene: dict) -> list[dict]:
    bins = {cat: [v * 100 for v in G.marker_centre(m)] for cat, m in SB.SORTING["category_bins"].items()}
    jobs = []
    for c in scene["cubes"]:
        home = [cat for cat, p in bins.items() if math.dist(c["map_cm"], p) <= SB.SORTING["in_place_cm"] + 0.5]
        bx, by = G.map_to_base(c["map_cm"][0] / 100, c["map_cm"][1] / 100)
        reach = math.hypot(bx, by)
        why = None
        if c["category"] is None:
            why = c["source"]
        elif home and home[0] == c["category"]:
            why = f"already on its bin ({c['category']})"
        elif plan_for(c) is None:
            why = f"{reach*100:.1f} cm from the base: no reachable grasp at pitch 130-170"
        elif c["coverage"] < 0.9:
            why = f"only {c['coverage']*100:.0f}% of it was in view; rescan with it nearer the middle"
        jobs.append(dict(cube=c, bin=SB.SORTING["category_bins"].get(c["category"]) if c["category"] else None,
                         skip=why))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", type=Path, help="use this scan directory instead of scanning")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import field_scan as SC
    with (G.nullcontext(None) if args.dry_run and args.scan else G.Session()) as sess:
        run = (args.scan.resolve() if args.scan else None) or SC.scan(session=sess)
        scene = build_scene(run)
        use_registration(scene["registration"])
        if not args.dry_run:
            scene = refine(run, scene, sess)
            use_registration(scene["registration"])
        jobs = decide(scene)
        for j in jobs:
            c = j["cube"]
            p = None if j["skip"] else plan_for(c)
            grasp = f" pitch {p['pitch']} hover +{p['hover_above_top_cm']:.1f} cm J5 {p['j5']}" if p else ""
            print(f"cube map ({c['map_cm'][0]:6.1f}, {c['map_cm'][1]:5.1f}) yaw {c['yaw_deg']:5.1f}{grasp}  top {c['colour'] or '-':6s} "
                  f"sticker {c['sticker'] and (c['sticker']['label'], c['sticker']['conf'])}  -> {c['category']} "
                  f"({c['source']})  {'SKIP: ' + j['skip'] if j['skip'] else 'bin ' + j['bin']}")
        print("registration:", scene["registration"])
        if args.dry_run:
            return
        occupied = [j["cube"]["map_cm"] for j in jobs]
        for j in jobs:
            if j["skip"]:
                continue
            target = [v * 100 for v in G.marker_centre(j["bin"])]
            if any(math.dist(p, target) <= SB.SORTING["occupied_cm"] and p != j["cube"]["map_cm"] for p in occupied):
                print("skip: bin", j["bin"], "occupied")
                continue
            plan = plan_for(j["cube"])
            if plan is None:
                print("skip: no reachable grasp for", j["cube"]["map_cm"])
                continue
            G.PLAN.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
            started = time.time()
            G.hover(sess)
            try:
                G.grasp(sess)
            except G.GripMissed as e:
                print("missed:", e)
                continue
            G.put(j["bin"], sess)
            occupied = [p for p in occupied if p != j["cube"]["map_cm"]] + [target]
            G.log("pipeline_sorted", cube=j["cube"]["map_cm"], bin=j["bin"], category=j["cube"]["category"],
                  source=j["cube"]["source"], seconds=round(time.time() - started))
        G.observe(sess)


if __name__ == "__main__":
    main()

"""Touch the top of every block the camera found, and report what the touch says about the model.

    python3 src/tasks/probe_blocks.py                 # scan, then probe each block (moves the arm)
    python3 src/tasks/probe_blocks.py --scan <dir>    # reuse a scan
    python3 src/tasks/probe_blocks.py --plan-only     # print where it would probe, move nothing

No block is gripped: the closed fingertips come down at the centre the vision pipeline predicts, and stop
where they meet something. Two numbers come out of that, and neither can be had from a picture:

  * **The height of the top face.** Predicted is `table_z + 3 cm`, from the fitted table height, the
    forward kinematics and the assumption that a block is 3 cm. A touch measures it. A standing
    disagreement is the model being wrong, in centimetres, and it is the same error the grasp descends
    with (it aims at `table_z + grasp_height_cm`).
  * **Whether the centre is even on the block.** Fingers that go all the way to the table found no block
    within about 1.5 cm of where the camera put it -- and closing them there is what flicks a block away
    instead of holding it (docs/05, 2026-09-27).

Downwards only. A block is not fixed to the table, so a sideways probe pushes it instead of feeling it;
the table underneath is what lets a downward touch read a height at all.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("pipeline", "tasks", "tools", "spatial", "vision"):
    sys.path.insert(0, str(ROOT / "src" / sub))

import console_grasp as G  # noqa: E402

START_ABOVE_CM = 1.5      # begin the descent this far above the predicted top face
FLOOR_ABOVE_CM = 0.4      # stop here above the table: no block, and the fingers stay clear of it
ON_BLOCK_CM = 1.0         # contact within this of the predicted top face counts as "on the block"
RUNS = ROOT / "data" / "runs" / "probe"


def probe_one(cube: dict, plan_for, session) -> dict:
    """Closed fingertips down at one block's predicted centre. Returns the measurement, or a reason."""
    plan = plan_for(cube)
    if plan is None:
        return dict(map_cm=cube["map_cm"], skipped="no reachable pose")
    tx, ty = plan["tip_target_cm"][0] / 100, plan["tip_target_cm"][1] / 100
    top = G.TABLE_Z + G.BLOCK
    started = time.time()

    G.check_path(G.status()["pose"], plan["hover"])
    session.pose(list(plan["hover"]) + [plan["j5"] or G.ROLL_ZERO])
    session.grip("grip")                          # closed: the forward-kinematics tip is the closed fingertip
    G.descend_to(session, tx, ty, top + START_ABOVE_CM / 100, plan["pitch"])
    got = G.probe_down(session, tx, ty, plan["pitch"],
                       from_z=top + START_ABOVE_CM / 100, to_z=G.TABLE_Z + FLOOR_ABOVE_CM / 100)
    try:
        session.rearm()                           # a contact may have made the console hold and disarm
    except G.ConsoleError:
        pass
    G.lift(session, plan["hover"])
    session.grip("open")

    predicted = round((top - G.TABLE_Z) * 100, 2)
    measured = got["above_table_cm"]
    return dict(map_cm=cube["map_cm"], yaw_deg=cube.get("yaw_deg"), category=cube.get("category"),
                pitch=plan["pitch"], aligned=plan.get("aligned"), contacted=got["contacted"],
                why=got["why"], predicted_top_cm=predicted, measured_top_cm=measured,
                error_cm=round(measured - predicted, 2),
                on_block=got["contacted"] and abs(measured - predicted) <= ON_BLOCK_CM,
                seconds=round(time.time() - started))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    import field_scan as SC
    import run as R
    G.require_empty_gripper()
    with (G.nullcontext(None) if args.plan_only and args.scan else G.Session()) as sess:
        scan = args.scan.resolve() if args.scan else SC.scan(session=sess)
        scene = R.build_scene(scan)
        R.use_registration(scene["registration"])
        print("registration:", scene["registration"])
        rows = []
        for cube in scene["cubes"]:
            if args.plan_only:
                plan = R.plan_for(cube)
                print(f"  {str(cube['category'] or '?'):11s} 地图 {cube['map_cm']} 朝向 {cube['yaw_deg']:5.1f}  "
                      f"{'俯角 %d 对齐 %s' % (plan['pitch'], plan['aligned']) if plan else '无可达姿态'}  "
                      f"预测顶面 {(G.TABLE_Z + G.BLOCK - G.TABLE_Z) * 100:.1f} cm")
                continue
            row = probe_one(cube, R.plan_for, sess)
            rows.append(row)
            if row.get("skipped"):
                print(f"  {str(row.get('category') or '?'):11s} {row['map_cm']}  跳过：{row['skipped']}")
            else:
                print(f"  {str(row['category'] or '?'):11s} {row['map_cm']} 俯角 {row['pitch']}  "
                      f"预测 {row['predicted_top_cm']:.2f} cm  实测 {row['measured_top_cm']:.2f} cm  "
                      f"差 {row['error_cm']:+.2f} cm  "
                      f"{'碰到块' if row['on_block'] else ('碰到桌面/未碰到' if not row['contacted'] else '碰到但高度不符')}"
                      f"  {row['seconds']} s")
        if not args.plan_only:
            G.observe(sess)

    if rows:
        errs = [r["error_cm"] for r in rows if r.get("contacted")]
        on = sum(bool(r.get("on_block")) for r in rows)
        print(f"\n{len(rows)} 块：碰到块 {on}，其余 {len(rows) - on}")
        if errs:
            print(f"顶面高度误差：中位 {statistics.median(errs):+.2f} cm，范围 {min(errs):+.2f} .. {max(errs):+.2f} cm")
            print("（系统性偏差说明桌面高度/正解/块高这套模型偏了，抓取正是按同一套模型下探的）")
        RUNS.mkdir(parents=True, exist_ok=True)
        out = RUNS / (time.strftime("%Y%m%d_%H%M%S") + ".json")
        out.write_text(json.dumps(dict(rows=rows, table_z_cm=round(G.TABLE_Z * 100, 2),
                                       grasp_height_cm=round(G.GRASP_HEIGHT * 100, 2)),
                                  ensure_ascii=False, indent=1))
        print("记录：", out)


if __name__ == "__main__":
    main()

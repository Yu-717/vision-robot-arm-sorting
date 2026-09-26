"""Whole-field scan with the wrist camera only (no phone), through the running console.

Five high views chosen from the calibrated camera model (docs/05, 2026-09-26) cover the far grasp band
(map y 10-17 cm, |x| <= 13 cm) with a 30 px image margin; three folded near views (camera looking down in
front of the platform, 1.1 cm clear of the platform box refitted 2026-09-25) add the near strip, so 91 % of
the map points a block can be grasped at are seen (the rest is the row right at the platform edge). At each view the arm must be at rest; the frame and the console's
official-model detections are taken from a board result computed after the stop.

    python3 src/pipeline/field_scan.py [run_dir]      # moves the arm; operator watching
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tasks"):
    sys.path.insert(0, str(ROOT / "src" / sub))

import console_grasp as G  # noqa: E402
from console_step import BASE, Session, status  # noqa: E402

SCAN_POSES = [[100, 125, 0, 0], [70, 125, 0, 0], [110, 105, 15, 0], [80, 120, 0, 0], [40, 90, 30, 0],
              [50, 105, 5, 0], [90, 105, 5, 0], [125, 105, 5, 0]]      # near views
RUNS = ROOT / "data" / "runs" / "pipeline"


def frame_at_rest(settle: float = 0.8, timeout: float = 10.0) -> tuple[bytes, dict]:
    """A board frame taken after the readback has been still for `settle` seconds. The console's board
    result for this pose is attached when one arrives (it may be an error: a view without enough markers
    is still a useful view; registration is done offline from all views)."""
    end = time.time() + timeout
    last = status()
    while time.time() < end:
        time.sleep(settle)
        now = status()
        if not now["busy"] and all(abs(a - b) <= 1 for a, b in zip(now["pose"], last["pose"]) if a is not None and b is not None):
            break
        last = now
    else:
        raise G.ConsoleError("the arm did not come to rest")
    jpeg = urllib.request.urlopen(f"{BASE}/api/frame", timeout=5).read()
    t0 = time.time()
    while time.time() - t0 < 8:   # the console's result for this pose (with its official-model detections)
        s = status()
        board = s["localization"].get("board") or {}
        if board.get("t", 0) > now["observed_at"] and board.get("servos") and \
                all(abs(a - b) <= 1 for a, b in zip(board["servos"], now["pose"]) if a is not None and b is not None):
            return jpeg, s
        time.sleep(0.4)
    return jpeg, now


def scan(run: Path | None = None, session: Session | None = None) -> Path:
    run = run or RUNS / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    views = []
    with G.lease(session) as sess:
        j5 = G.ROLL_ZERO
        for i, pose in enumerate(SCAN_POSES):
            G.move_to(sess, pose, j5)
            jpeg, s = frame_at_rest()
            name = f"view_{i:02d}.jpg"
            (run / name).write_bytes(jpeg)
            board = s["localization"].get("board") or {}
            views.append(dict(image=name, readback=s["pose"], command=pose + [j5], markers=board.get("markers"),
                              official=board.get("official", []), t=time.time()))
            G.log("scan_view", run=str(run.relative_to(ROOT)), view=i, pose=s["pose"], markers=board.get("markers"),
                  official=[(d.get("label"), d.get("conf")) for d in board.get("official", [])])
    (run / "views.json").write_text(json.dumps(views, indent=1, ensure_ascii=False))
    return run


if __name__ == "__main__":
    print(scan(Path(sys.argv[1]) if len(sys.argv) > 1 else None))

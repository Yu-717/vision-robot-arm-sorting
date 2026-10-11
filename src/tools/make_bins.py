# /// script
# requires-python = ">=3.9"
# dependencies = ["opencv-python>=4.8,<5", "numpy", "pillow"]
# ///
"""Generate the acrylic bin design: cut drawings, DXF, marker stickers and layout JSON.

  uv run src/tools/make_bins.py --thickness-mm 2.0

One bin is a base plate with four identical walls glued on top in a pinwheel (each wall
butts against the next), so the floor and the marker flange are the same plate and every
marker sits at the plate thickness above the table. Measure the sheet with calipers
before ordering: wall length = inner + thickness.

Bin frame: origin at the centre of the opening on the plate's top face, +Y towards the
marker flange, X to the right when looking along +Y, millimetres in the drawings and
metres in the JSON.

Everything written here is a design value. Nothing has been cut or measured yet; the
placement page only checks the planned release points against the upstream link model
(src/spatial/dofbot_kinematics.py) and the soft limits in config/arm_limits.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "spatial"))
import dofbot_kinematics as kin  # noqa: E402

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
DICTIONARY = "DICT_5X5_50"  # the map and the table fiducials use DICT_4X4_50, the ChArUco board its ids 0-16
CATEGORIES = [("recyclable", "可回收物"), ("kitchen", "厨余垃圾"), ("hazardous", "有害垃圾"), ("other", "其他垃圾")]
BLOCK_MM = 30.0
A3 = (420.0, 297.0)


class Canvas:
    """A page in millimetres, y up, drawn at `scale` drawing-mm per real mm."""

    def __init__(self, size_mm: tuple[float, float], dpi: int, scale: float = 1.0,
                 origin_mm: tuple[float, float] = (0.0, 0.0)):
        self.k, self.scale, self.origin, self.h = dpi / 25.4, scale, origin_mm, size_mm[1]
        self.image = Image.new("RGB", (round(size_mm[0] * self.k), round(size_mm[1] * self.k)), "white")
        self.draw = ImageDraw.Draw(self.image)

    def px(self, x: float, y: float) -> tuple[int, int]:
        return (round((self.origin[0] + x * self.scale) * self.k),
                round((self.h - self.origin[1] - y * self.scale) * self.k))

    def font(self, mm: float) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(FONT, max(8, round(mm * self.k)))

    def line(self, a, b, fill="black", width_mm: float = 0.25, dash_mm: float = 0.0) -> None:
        w = max(1, round(width_mm * self.k))
        if not dash_mm:
            self.draw.line([self.px(*a), self.px(*b)], fill=fill, width=w)
            return
        length = math.dist(a, b) * self.scale
        n = max(1, int(length / dash_mm))
        for i in range(0, n, 2):
            p = [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in (i, min(i + 1, n))]
            self.draw.line([self.px(*p[0]), self.px(*p[1])], fill=fill, width=w)

    def poly(self, points, outline="black", fill=None, width_mm: float = 0.25, dash_mm: float = 0.0) -> None:
        if fill:
            self.draw.polygon([self.px(*p) for p in points], fill=fill)
        for a, b in zip(points, points[1:] + points[:1]):
            self.line(a, b, outline, width_mm, dash_mm)

    def rect(self, x: float, y: float, w: float, h: float, **kw) -> None:
        self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], **kw)

    def text(self, x: float, y: float, s: str, mm: float = 3.0, fill="black", anchor: str = "mm") -> None:
        self.draw.text(self.px(x, y), s, fill=fill, font=self.font(mm), anchor=anchor)

    def dim(self, a, b, off: float, label: str | None = None, mm: float = 2.8) -> None:
        """Dimension line between a and b, shifted `off` real-mm to the left of a->b."""
        d = math.dist(a, b)
        n = (-(b[1] - a[1]) / d, (b[0] - a[0]) / d)
        a2, b2 = (a[0] + n[0] * off, a[1] + n[1] * off), (b[0] + n[0] * off, b[1] + n[1] * off)
        blue = (40, 80, 200)
        for p, q in ((a, a2), (b, b2)):
            self.line(p, (q[0] + n[0] * 1.5 / self.scale * (1 if off > 0 else -1),
                          q[1] + n[1] * 1.5 / self.scale * (1 if off > 0 else -1)), blue, 0.12)
        self.line(a2, b2, blue, 0.15)
        t = 1.2 / self.scale
        u = ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
        for p in (a2, b2):
            self.line((p[0] - (u[0] + n[0]) * t, p[1] - (u[1] + n[1]) * t),
                      (p[0] + (u[0] + n[0]) * t, p[1] + (u[1] + n[1]) * t), blue, 0.25)
        shift = 2.4 / self.scale * (1 if off > 0 else -1)
        mid = ((a2[0] + b2[0]) / 2 + n[0] * shift, (a2[1] + b2[1]) / 2 + n[1] * shift)
        self.text(*mid, label or f"{d:g}", mm, blue)

    def marker(self, mid: int, cx: float, cy: float, side: float) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
        x0, y0 = self.px(cx - side / 2, cy + side / 2)
        x1, _ = self.px(cx + side / 2, cy + side / 2)
        self.image.paste(Image.fromarray(cv2.aruco.generateImageMarker(dictionary, mid, x1 - x0, borderBits=1)), (x0, y0))


class Design:
    def __init__(self, a: argparse.Namespace):
        self.t, self.inner, self.wall_h = a.thickness_mm, a.inner_mm, a.wall_mm
        self.margin, self.flange, self.side = a.margin_mm, a.flange_mm, a.marker_mm
        self.outer = self.inner + 2 * self.t
        self.wall_len = self.inner + self.t
        self.base_w = self.outer + 2 * self.margin
        self.base_d = self.outer + self.margin + self.flange
        self.y0 = -(self.outer / 2 + self.margin)          # plate's near edge in the bin frame
        self.y1 = self.outer / 2 + self.flange             # plate's flange edge
        self.flange_y = self.outer / 2 + self.flange / 2   # marker centres on the flange
        self.flange_x = a.marker_pitch_mm / 2

    def marker_centres(self) -> dict[str, tuple[float, float]]:
        return {"floor": (0.0, 0.0), "flange_left": (-self.flange_x, self.flange_y),
                "flange_right": (self.flange_x, self.flange_y)}

    def marker_ids(self, number: int) -> dict[str, int]:
        return {"floor": 10 * number, "flange_left": 10 * number + 1, "flange_right": 10 * number + 2}

    def walls(self) -> list[list[tuple[float, float]]]:
        """Footprints of the four pinwheel walls in the bin frame."""
        h, t = self.inner / 2, self.t
        first = [(-h - t, -h - t), (h, -h - t), (h, -h), (-h - t, -h)]  # near wall, running +X
        return [[(x * c - y * s, x * s + y * c) for x, y in first]
                for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1))]

    def plate(self) -> list[tuple[float, float]]:
        w = self.base_w / 2
        return [(-w, self.y0), (w, self.y0), (w, self.y1), (-w, self.y1)]


def draw_bin_top(c: Canvas, d: Design, ox: float, oy: float, number: int | None = None, hatch: bool = True) -> None:
    def at(points):
        return [(ox + x, oy + y) for x, y in points]
    c.poly(at(d.plate()), width_mm=0.4)
    for wall in d.walls():
        c.poly(at(wall), fill=(205, 225, 245) if hatch else None, width_mm=0.25)
    for name, (x, y) in d.marker_centres().items():
        if number is None:
            c.rect(ox + x - d.side / 2, oy + y - d.side / 2, d.side, d.side, outline=(120, 120, 120), dash_mm=1.0)
        else:
            c.marker(d.marker_ids(number)[name], ox + x, oy + y, d.side)


def page_parts(d: Design) -> Image.Image:
    c = Canvas(A3, 150)
    c.text(10, 287, f"亚克力垃圾桶 v1 · 零件与装配图（1:1，单位 mm）· 板厚 t = {d.t:g}", 5, anchor="lm")
    c.text(10, 279, "设计稿，尚未加工与实测。下单前用卡尺量实际板厚并用 --thickness-mm 重新生成。", 3.2, (150, 0, 0), "lm")

    # base plate
    ox, oy = 75, 150 - (d.y0 + d.y1) / 2
    c.text(ox, oy + d.y1 + 22, "① 底板 ×4（每桶 1 块）", 4)
    draw_bin_top(c, d, ox, oy, hatch=False)
    for wall in d.walls():
        c.poly([(ox + x, oy + y) for x, y in wall], outline=(150, 150, 150), dash_mm=1.2, width_mm=0.15)
    w = d.base_w / 2
    c.dim((ox - w, oy + d.y0), (ox + w, oy + d.y0), -9)
    c.dim((ox + w, oy + d.y0), (ox + w, oy + d.y1), -10)
    c.dim((ox - w, oy + d.outer / 2), (ox - w, oy + d.y1), 9, f"{d.flange:g}")
    c.dim((ox - w, oy + d.y0), (ox - w, oy - d.outer / 2), 9, f"{d.margin:g}")
    c.dim((ox - d.flange_x, oy + d.flange_y), (ox + d.flange_x, oy + d.flange_y), d.flange / 2 + 6,
          f"码中心距 {2 * d.flange_x:g}")
    c.dim((ox - d.inner / 2, oy - 24), (ox + d.inner / 2, oy - 24), -4, f"内口 {d.inner:g}")
    c.text(ox, oy + 12, "桶底码", 2.6, (120, 120, 120))
    c.text(ox, oy + d.flange_y, f"{d.side:g}", 2.4, (120, 120, 120))
    c.text(ox, oy + d.y0 - 20, "虚线 = 侧板落位线（可让商家浅雕，非切割）", 2.8, (110, 110, 110))

    # wall
    wx, wy = 160, 205
    c.text(wx + d.wall_len / 2, wy + d.wall_h + 12, "② 侧板 ×16（每桶 4 块，全部相同）", 4)
    c.rect(wx, wy, d.wall_len, d.wall_h, width_mm=0.4, fill=(205, 225, 245))
    c.dim((wx, wy), (wx + d.wall_len, wy), -8, f"{d.wall_len:g}（= 内口 {d.inner:g} + t）")
    c.dim((wx + d.wall_len, wy), (wx + d.wall_len, wy + d.wall_h), -8)

    # assembly, top
    ax, ay = 355, 150 - (d.y0 + d.y1) / 2
    c.text(ax, ay + d.y1 + 22, "装配俯视（风车式对接）", 4)
    draw_bin_top(c, d, ax, ay, number=1)
    h = d.inner / 2
    c.dim((ax - h - d.t, ay - h - d.t), (ax + h + d.t, ay - h - d.t), -(d.margin + 9), f"外廓 {d.outer:g}")
    c.poly([(ax - BLOCK_MM / 2 + 18, ay - BLOCK_MM / 2 - 14), (ax + BLOCK_MM / 2 + 18, ay - BLOCK_MM / 2 - 14),
            (ax + BLOCK_MM / 2 + 18, ay + BLOCK_MM / 2 - 14), (ax - BLOCK_MM / 2 + 18, ay + BLOCK_MM / 2 - 14)],
           outline=(200, 90, 0), dash_mm=1.0)
    c.text(ax + 18, ay - 14, "3 cm 块", 2.4, (200, 90, 0))
    c.text(ax, ay + d.y0 - 20, "每块侧板一端顶住下一块的内面，四块首尾相接", 2.8, (110, 110, 110))

    # assembly, side
    sx, sy = 160, 110
    c.text(sx + d.base_d / 2, sy + d.t + d.wall_h + 12, "装配侧视", 4)
    c.rect(sx, sy, d.base_d, d.t, width_mm=0.4, fill=(225, 225, 225))
    c.rect(sx + d.margin, sy + d.t, d.outer, d.wall_h, width_mm=0.3)
    for x in (sx + d.margin, sx + d.margin + d.outer - d.t):
        c.rect(x, sy + d.t, d.t, d.wall_h, width_mm=0.3, fill=(205, 225, 245))
    c.dim((sx, sy), (sx + d.base_d, sy), -8)
    c.dim((sx, sy), (sx, sy + d.t + d.wall_h), 8, f"总高 {d.t + d.wall_h:g}")
    c.dim((sx + d.base_d, sy), (sx + d.base_d, sy + d.t), -8, f"码面高 = t = {d.t:g}")
    c.rect(sx + d.margin + d.outer + d.flange / 2 - d.side / 2, sy + d.t, d.side, 0.6, fill="black")
    c.text(sx + d.margin + d.outer + d.flange / 2, sy + d.t + 5, "定位码", 2.6)
    c.rect(sx + d.margin + d.t + 20, sy + d.t, BLOCK_MM, BLOCK_MM, outline=(200, 90, 0), dash_mm=1.0)

    notes = ["工艺：透明浇铸亚克力；激光切割或勾刀划线掰断均可（全部为直边矩形）。",
             "粘接：亚克力专用胶（溶剂型）沿内角渗入；先粘侧板成框，再落到底板落位线上。",
             "贴纸：侧板粘好后再贴。桶底贴纸落入框内四边自定位，耳板贴纸靠底板三条外边对齐。",
             "底面贴防滑垫或用蓝丁胶固定在桌面；码面高度按 t + 贴纸厚度写入配置。"]
    for i, s in enumerate(notes):
        c.text(150, 62 - i * 7, s, 3.0, anchor="lm")
    return c.image


def nest(d: Design, sheet: tuple[float, float], bins: int) -> list[tuple[str, float, float, float, float]]:
    """Guillotine strips with shared cut lines: plates lying down first, walls in what is left."""
    sw, sh = sheet
    out, y, walls_left, plates_left = [], 0.0, 4 * bins, bins
    per_row = int(sw // d.base_d)
    if not per_row:
        raise SystemExit("stock sheet narrower than one base plate")
    while plates_left:
        n = min(per_row, plates_left)
        for i in range(n):
            out.append(("底板", i * d.base_d, y, d.base_d, d.base_w))
        x = n * d.base_d
        while walls_left and x + d.wall_h <= sw and d.wall_len <= d.base_w:
            out.append(("侧板", x, y, d.wall_h, d.wall_len))
            x, walls_left = x + d.wall_h, walls_left - 1
        y, plates_left = y + d.base_w, plates_left - n
    while walls_left:
        n = min(int(sw // d.wall_h), walls_left)
        for i in range(n):
            out.append(("侧板", i * d.wall_h, y, d.wall_h, d.wall_len))
        y, walls_left = y + d.wall_len, walls_left - n
    if y > sh:
        raise SystemExit(f"parts need {y:g} mm of a {sw:g} x {sh:g} mm sheet; pass a larger --sheet-mm")
    return out


def page_sheet(d: Design, sheet: tuple[float, float], parts: list) -> Image.Image:
    scale = min(0.62, 250 / sheet[1], 250 / sheet[0])
    c = Canvas(A3, 150, scale, (60, 20))
    used = max(p[2] + p[4] for p in parts)
    title = Canvas(A3, 150)
    title.image = c.image
    title.draw = c.draw
    title.text(10, 287, f"切割排版参考 · 原板 {sheet[0]:g} × {sheet[1]:g} mm，板厚 {d.t:g} · 图面比例 {scale:.2f}", 5, anchor="lm")
    title.text(10, 279, "相邻零件共用切线（通裁排法，适合手工划线；激光切割时切缝约 0.2 mm 可忽略）。", 3.2, anchor="lm")
    c.rect(0, 0, *sheet, width_mm=0.5, fill=(245, 245, 245))
    for i, (name, x, y, w, h) in enumerate(parts):
        c.rect(x, y, w, h, width_mm=0.3, fill=(255, 255, 255) if name == "底板" else (205, 225, 245))
        c.text(x + w / 2, y + h / 2 + (5 if w > 60 else 0), name, 3.0)
        c.text(x + w / 2, y + h / 2 - (5 if w > 60 else 9), f"{w:g}×{h:g}" if w > 60 else f"{w:g}\n×\n{h:g}", 2.4, (90, 90, 90))
    c.dim((0, 0), (sheet[0], 0), -12 / scale)
    c.dim((0, 0), (0, sheet[1]), 12 / scale)
    c.dim((sheet[0], 0), (sheet[0], used), -12 / scale, f"用料高 {used:g}")
    rows = [("零件", "尺寸 (mm)", "数量"), ("① 底板", f"{d.base_w:g} × {d.base_d:g}", f"{sum(p[0] == '底板' for p in parts)}"),
            ("② 侧板", f"{d.wall_len:g} × {d.wall_h:g}", f"{sum(p[0] == '侧板' for p in parts)}")]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            title.text(285 + j * 42, 240 - i * 10, cell, 3.6, anchor="lm")
    for i, s in enumerate(["通裁顺序：先横向裁出各条带，", "再把每条带竖向裁成零件。",
                           "商家代切时直接给上表尺寸和数量，", "或发 DXF（每个零件各一份轮廓）。"]):
        title.text(285, 195 - i * 8, s, 3.2, (70, 70, 70), "lm")
    return c.image


def placements(spec: list[str]) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in s.split(",")) for s in spec]


def bin_pose(r_cm: float, theta_deg: float) -> tuple[float, float, float]:
    """Centre (mm, base frame) and the direction of the bin's +Y (radially outwards)."""
    th = math.radians(theta_deg)
    return r_cm * 10 * math.cos(th), r_cm * 10 * math.sin(th), theta_deg


def to_base(pose: tuple[float, float, float], points) -> list[tuple[float, float]]:
    cx, cy, th = pose
    c, s = math.cos(math.radians(th - 90)), math.sin(math.radians(th - 90))
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in points]


def release_check(d: Design, pose, arm: dict, clearance_mm: float) -> dict:
    """Release point above the bin centre in the upstream link model, soft limits, pitch 170-130."""
    table = arm["keep_out"]["table_surface_z_m"]
    grasp = arm["gripper"]["grasp_height_cm"] / 100
    z = table + (d.t + d.wall_h + clearance_mm) / 1000 + grasp
    limits = {int(k): v for k, v in arm["soft_limits_deg"].items()}
    found = kin.inverse_within(pose[0] / 1000, pose[1] / 1000, z, limits, tuple(range(170, 129, -5)))
    return {"tip_z_base_cm": round(z * 100, 2), "solution": found}


def page_layout(d: Design, poses: list, checks: list, arm: dict, map_cfg: dict) -> Image.Image:
    c = Canvas(A3, 150, 0.52, (210, 100))
    t = Canvas(A3, 150)
    t.image, t.draw = c.image, c.draw
    t.text(10, 287, "摆放方案与可达性核对（俯视，基座坐标系，单位 cm）· 规划值，未实测", 5, anchor="lm")
    dy = map_cfg["placement"]["base_from_map"]["dy"] * 1000
    c.rect(-148.5, dy, 297, 210, outline=(120, 120, 120), fill=(250, 250, 240))
    for corners in map_cfg["layout_m"].values():
        xs, ys = [p[0] * 1000 for p in corners], [p[1] * 1000 + dy for p in corners]
        c.rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), fill=(90, 90, 90), outline=(90, 90, 90))
    c.text(0, dy + 204, "A4 地图 map_a4_v2", 2.6, (120, 120, 120))

    def arc(r, a0, a1, **kw):
        pts = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in np.arange(a0, a1 + 0.1, 2)]
        for p, q in zip(pts, pts[1:]):
            c.line(p, q, **kw)
    pr = arm["keep_out"]["platform_radius_m"] * 1000
    arc(pr, 0, 360, fill="black", width_mm=0.4)
    c.text(0, 0, "底座平台", 2.6)
    lo, hi = arm["soft_limits_deg"]["1"]
    for r in (130, 280):
        arc(r, lo, hi, fill=(0, 140, 0), width_mm=0.3)
    for r in (180, 240):
        arc(r, lo, hi, fill=(0, 140, 0), width_mm=0.15)
    for a in (lo, hi):
        c.line((130 * math.cos(math.radians(a)), 130 * math.sin(math.radians(a))),
               (300 * math.cos(math.radians(a)), 300 * math.sin(math.radians(a))), (0, 140, 0), 0.3, dash_mm=1.5)
    c.text(0, 287, "可达环带 13–28 cm（细线：容器中心建议 18–24 cm）", 2.6, (0, 120, 0))
    for i, (pose, chk) in enumerate(zip(poses, checks)):
        ok = chk["solution"] is not None
        c.poly(to_base(pose, d.plate()), width_mm=0.4, fill=(255, 255, 255))
        for wall in d.walls():
            c.poly(to_base(pose, wall), fill=(150, 195, 240) if ok else (240, 150, 150), width_mm=0.15)
        for x, y in to_base(pose, list(d.marker_centres().values())):
            c.rect(x - d.side / 2, y - d.side / 2, d.side, d.side, fill="black")
        lx, ly = to_base(pose, [(0, -18)])[0]
        c.text(lx, ly, f"{i + 1}", 4.0, (200, 0, 0))
    rows = [("桶", "类别", "中心 r / J1", "中心 x, y", "松手指尖 z", "逆解（俯角 / 舵机 1–4）")]
    for i, (pose, chk) in enumerate(zip(poses, checks)):
        sol = chk["solution"]
        rows.append((f"{i + 1}", CATEGORIES[i % 4][1], f"{math.hypot(pose[0], pose[1]) / 10:.1f} / {pose[2]:.0f}°",
                     f"{pose[0] / 10:.1f}, {pose[1] / 10:.1f}", f"{chk['tip_z_base_cm']:.1f}",
                     f"{sol['pitch_deg']}° / {sol['servos']}" if sol else "无解"))
    xs = [12, 24, 50, 82, 112, 140]
    for i, row in enumerate(rows):
        for x, cell in zip(xs, row):
            t.text(x, 44 - i * 6.5, cell, 3.0, anchor="lm")
    t.text(12, 8, "逆解用上游连杆模型和软限位，只说明几何上可行；实机须按硬件安全边界小步验证后再写入结论。", 2.8, (150, 0, 0), "lm")
    return c.image


def page_stickers(d: Design, bins: int) -> tuple[Image.Image, dict]:
    c = Canvas((210.0, 297.0), 600)
    c.text(105, 289, f"垃圾桶定位码贴纸 · A4 · 必须按 100%/实际尺寸打印 · {DICTIONARY} · 码边长 {d.side:g} mm", 3.4)
    c.text(105, 283.5, "哑光不干胶纸打印，沿灰线裁下。耳板条贴底板耳板（外边与两侧边对齐），方片落入桶底。", 3.0, (90, 90, 90))
    grey, layout = (150, 150, 150), {}
    strip_h, floor = d.flange - 1.0, d.inner - 1.0
    for i in range(bins):
        number, (key, name) = i + 1, CATEGORIES[i % 4]
        ids = d.marker_ids(number)
        col, row = i % 2, i // 2
        # flange strip: its top edge is the plate's outer edge, so marker centres sit flange/2 below it
        x0, y1 = 6 + col * (d.base_w + 6), 275 - row * (strip_h + 6)
        c.rect(x0, y1 - strip_h, d.base_w, strip_h, outline=grey, width_mm=0.1)
        cx, cy = x0 + d.base_w / 2, y1 - d.flange / 2
        c.marker(ids["flange_left"], cx - d.flange_x, cy, d.side)
        c.marker(ids["flange_right"], cx + d.flange_x, cy, d.side)
        c.text(cx, cy + 4.5, f"{number}", 7.0)
        c.text(cx, cy - 5.0, name, 4.2)
        c.text(x0 + d.base_w / 2, y1 - 1.6, "▲ 此边对齐底板外边", 1.8, grey)
        # floor square
        fx, fy = 12 + col * (floor + 14), 275 - ((bins + 1) // 2) * (strip_h + 6) - 6 - (row + 1) * (floor + 6)
        c.rect(fx, fy, floor, floor, outline=grey, width_mm=0.1)
        c.marker(ids["floor"], fx + floor / 2, fy + floor / 2, d.side)
        c.text(fx + floor / 2, fy + floor - 6, f"▲ 朝耳板 · {number} {name}", 2.6, grey)
        layout[str(number)] = {"category": key, "name": name, "ids": ids}
    c.line((20, 8), (120, 8), width_mm=0.3)
    for k in range(11):
        c.line((20 + 10 * k, 8), (20 + 10 * k, 10.5), width_mm=0.3)
    c.text(126, 8.5, "100 mm 校验尺", 3.0, anchor="lm")
    return c.image, layout


def write_dxf(path: Path, d: Design, bins: int) -> None:
    """One outline per part (R12 LINE entities, millimetres); wall seats on a separate engraving layer."""
    lines: list[tuple[str, tuple, tuple]] = []

    def loop(layer, pts):
        lines.extend((layer, a, b) for a, b in zip(pts, pts[1:] + pts[:1]))
    for i in range(bins):
        ox, oy = i * (d.base_w + 10) + d.base_w / 2, -d.y0
        loop("CUT", [(ox + x, oy + y) for x, y in d.plate()])
        for wall in d.walls():
            loop("ENGRAVE_OPTIONAL", [(ox + x, oy + y) for x, y in wall])
    for k in range(4 * bins):
        x, y = (k % 4) * (d.wall_len + 10), d.base_d + 10 + (k // 4) * (d.wall_h + 10)
        loop("CUT", [(x, y), (x + d.wall_len, y), (x + d.wall_len, y + d.wall_h), (x, y + d.wall_h)])
    body = ["0", "SECTION", "2", "ENTITIES"]
    for layer, a, b in lines:
        body += ["0", "LINE", "8", layer, "10", f"{a[0]:.3f}", "20", f"{a[1]:.3f}", "11", f"{b[0]:.3f}", "21", f"{b[1]:.3f}"]
    path.write_text("\n".join(body + ["0", "ENDSEC", "0", "EOF"]) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="bins_acrylic_v1")
    p.add_argument("--thickness-mm", type=float, default=2.0, help="measured sheet thickness")
    p.add_argument("--inner-mm", type=float, default=80.0, help="square opening")
    p.add_argument("--wall-mm", type=float, default=40.0, help="wall height above the floor")
    p.add_argument("--margin-mm", type=float, default=5.0, help="plate rim outside the walls on three sides")
    p.add_argument("--flange-mm", type=float, default=32.0, help="plate rim on the marker side")
    p.add_argument("--marker-mm", type=float, default=20.0)
    p.add_argument("--marker-pitch-mm", type=float, default=50.0, help="distance between the two flange markers")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--sheet-mm", type=float, nargs=2, default=[300.0, 400.0])
    p.add_argument("--place", nargs="+", default=["22,24", "22,55", "22,125", "22,156"],
                   help="planned bin centres as r_cm,theta_deg (theta = servo-1 angle); flange points outwards")
    p.add_argument("--release-clearance-mm", type=float, default=10.0, help="block bottom above the wall top at release")
    p.add_argument("--preview-dir", type=Path, help="also write each page as a PNG here")
    a = p.parse_args()
    d = Design(a)

    arm = json.loads((ROOT / "config" / "arm_limits.json").read_text())
    map_cfg = json.loads((ROOT / "config" / "map_a4_v2.json").read_text())
    poses = [bin_pose(*rt) for rt in placements(a.place)]
    checks = [release_check(d, pose, arm, a.release_clearance_mm) for pose in poses]
    parts = nest(d, tuple(a.sheet_mm), a.bins)
    stickers, bins = page_stickers(d, a.bins)

    out = ROOT / "docs" / "bins"
    out.mkdir(exist_ok=True)
    pages = [page_parts(d), page_sheet(d, tuple(a.sheet_mm), parts), page_layout(d, poses, checks, arm, map_cfg)]
    pages[0].save(out / f"{a.name}_drawings.pdf", save_all=True, append_images=pages[1:], resolution=150)
    stickers.save(out / f"{a.name}_stickers_A4.pdf", resolution=600)
    write_dxf(out / f"{a.name}_cut.dxf", d, a.bins)

    h = d.side / 2
    for number, entry in bins.items():
        entry["corners_m"] = {
            str(entry["ids"][k]): [[round(v / 1000, 5) for v in pt] for pt in
                                   ((x - h, y + h), (x + h, y + h), (x + h, y - h), (x - h, y - h))]
            for k, (x, y) in d.marker_centres().items()}
        pose, chk = poses[int(number) - 1], checks[int(number) - 1]
        entry["planned_centre_base_m"] = [round(pose[0] / 1000, 4), round(pose[1] / 1000, 4)]
        entry["planned_flange_direction_deg"] = pose[2]
        entry["planned_release"] = chk
    cfg = {
        "_status": f"design values from src/tools/make_bins.py for docs/bins/{a.name}_*; not cut, not measured, "
                   "placements and release solutions are plans from the upstream link model only",
        "dictionary": DICTIONARY, "side_m": d.side / 1000,
        "bin_frame": "origin at the centre of the opening on the plate's top face; +Y towards the marker flange, metres",
        "corner_order": "ArUco order TL, TR, BR, BL with the marker top pointing to +Y",
        "marker_plane_above_table_m": d.t / 1000,
        "geometry_mm": {"thickness": d.t, "inner": d.inner, "outer": d.outer, "wall_height": d.wall_h,
                        "wall_length": d.wall_len, "plate": [d.base_w, d.base_d], "plate_y_range": [d.y0, d.y1],
                        "rim_top_above_table": d.t + d.wall_h},
        "bins": bins,
    }
    (ROOT / "config" / f"{a.name}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    if a.preview_dir:
        for i, page in enumerate(pages + [stickers]):
            page.save(a.preview_dir / f"{a.name}_preview_{i + 1}.png")

    # self-check: every sticker marker is found at roughly camera resolution, at its designed spacing
    grey = cv2.resize(np.array(stickers.convert("L")), None, fx=0.12, fy=0.12, interpolation=cv2.INTER_AREA)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()).detectMarkers(grey)
    found = {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.ravel(), corners)} if ids is not None else {}
    want = sorted(v for e in bins.values() for v in e["ids"].values())
    mm_per_px = 25.4 / (600 * 0.12)
    pitch = [np.linalg.norm(found[e["ids"]["flange_left"]] - found[e["ids"]["flange_right"]]) * mm_per_px
             for e in bins.values() if e["ids"]["flange_left"] in found and e["ids"]["flange_right"] in found]
    print(f"parts: plate {d.base_w:g} x {d.base_d:g} x{a.bins}, wall {d.wall_len:g} x {d.wall_h:g} x{4 * a.bins}, "
          f"sheet use {max(q[2] + q[4] for q in parts):g} of {a.sheet_mm[1]:g} mm")
    print(f"self-check: found {len(set(found) & set(want))}/{len(want)} sticker ids, "
          f"flange pitch {min(pitch):.2f}-{max(pitch):.2f} mm (design {2 * d.flange_x:g})")
    polys = [np.array(to_base(pose, d.plate()), np.float32) for pose in poses]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if cv2.intersectConvexConvex(polys[i], polys[j])[0] > 1.0:
                print(f"WARNING: plates of bins {i + 1} and {j + 1} overlap")
    for i, (pose, chk) in enumerate(zip(poses, checks)):
        print(f"bin {i + 1}: centre ({pose[0] / 10:.1f}, {pose[1] / 10:.1f}) cm, release tip z {chk['tip_z_base_cm']} cm -> {chk['solution']}")


if __name__ == "__main__":
    main()

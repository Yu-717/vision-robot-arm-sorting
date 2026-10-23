// 三维状态视图（F1-3）：骨架点来自服务端算好的正运动学 status.arm.chain_cm（厘米），
// 这里只负责旋转、投影和画。刻意不在前端重算运动学——图上看到的位置，和控制台拿去做
// 禁入区检查的必须是同一份数。世界坐标沿用渗出坐标系：x 右（臂的右侧）、y 远离臂、z 上。

const CHAIN = ["base", "shoulder", "elbow", "wrist", "servo5", "tip"];
const LABELS = { tip: "末端", wrist: "腕", elbow: "肘", servo5: "腕旋" };
const DIST = 40;            // 相机到原点的距离，厘米
const rad = (d) => (d * Math.PI) / 180;

export function createSceneView(canvas) {
  const ctx = canvas.getContext("2d");
  let yaw = 38, pitch = 20, zoom = 1;
  let drag = null;
  let chain = null;

  const rotate = ([x, y, z]) => {
    const cy = rad(yaw), cp = rad(pitch);
    const rx = x * Math.cos(cy) - y * Math.sin(cy);
    const ry = x * Math.sin(cy) + y * Math.cos(cy);
    return [rx, ry * Math.cos(cp) - z * Math.sin(cp), ry * Math.sin(cp) + z * Math.cos(cp)];
  };

  // 相机空间 (右, 深, 上) -> 画布坐标；深度为第二分量
  const project = (p) => {
    const [rx, depth, up] = rotate(p);
    const w = canvas.clientWidth || 360, h = canvas.clientHeight || 260;
    const f = Math.min(w, h) * 1.4 * zoom;
    const d = Math.max(6, DIST + depth);
    return [w / 2 + (rx * f) / d, h / 2 + 34 - (up * f) / d, d];
  };

  const line = (from, to, width, colour) => {
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to); ctx.stroke();
  };

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 360, h = canvas.clientHeight || 260;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b141b"; ctx.fillRect(0, 0, w, h);

    // 桌面网格：±20 cm，每 5 cm 一条，给姿态一个参照
    ctx.lineWidth = 1;
    for (let v = -20; v <= 20; v += 5) {
      const a = project([v, -20, 0]), b = project([v, 20, 0]);
      const c = project([-20, v, 0]), d = project([20, v, 0]);
      ctx.strokeStyle = v === 0 ? "#2f4859" : "#1c2b36";
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
    }

    if (!chain) {
      ctx.fillStyle = "#8fa5b4"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
      ctx.fillText("还没有回读：接控制台后这里显示真实姿态", w / 2, h / 2);
      return;
    }

    const pts = {};
    for (const key of CHAIN) {
      if (!chain[key]) continue;
      const p = project(chain[key]);
      pts[key] = p;
    }
    if (!pts.shoulder) return;

    // 骨架：按深度排序画，远的先画
    const order = CHAIN.filter((k) => pts[k]).sort((a, b) => pts[b][2] - pts[a][2]);
    const seg = (a, b) => pts[a] && pts[b] && line(pts[a].slice(0, 2), pts[b].slice(0, 2), 6, "#4fb3d9");

    line(pts.base.slice(0, 2), pts.shoulder.slice(0, 2), 7, "#33586b");   // 底座立柱
    seg("shoulder", "elbow");
    seg("elbow", "wrist");
    seg("wrist", "servo5");
    line(pts.servo5.slice(0, 2), pts.tip.slice(0, 2), 4, "#8fa5b4");     // 夹爪

    for (const key of order) {
      const [sx, sy] = pts[key];
      const tip = key === "tip";
      ctx.beginPath(); ctx.arc(sx, sy, tip ? 4.5 : 3.5, 0, 7);
      ctx.fillStyle = tip ? "#5fc48a" : "#e3edf3"; ctx.fill();
      if (LABELS[key]) {
        ctx.fillStyle = "#8fa5b4"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
        ctx.fillText(LABELS[key], sx + 7, sy + 3);
      }
    }

    ctx.fillStyle = "#5a7180"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    ctx.fillText("拖动旋转 · 滚轮缩放", w - 6, h - 6);
  }

  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw, pitch };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    yaw = drag.yaw + (e.clientX - drag.x) * 0.6;
    pitch = Math.max(-80, Math.min(80, drag.pitch + (e.clientY - drag.y) * 0.4));
    draw();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom = Math.max(0.4, Math.min(3.5, zoom * (1 - e.deltaY * 0.0012)));
    draw();
  }, { passive: false });
  window.addEventListener("resize", draw);

  return {
    /** @param {{base,shoulder,elbow,wrist,servo5,tip}|null} value 服务端给的骨架点，厘米 */
    set(value) {
      const next = value && Object.keys(value).length ? value : null;
      if (next !== chain) { chain = next; draw(); }
    },
    draw,
  };
}

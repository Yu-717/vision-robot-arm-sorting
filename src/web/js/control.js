// F1-4 模式切换（W1）与 F1-3 控制面板（W2）。
// 指令一律经 POST /api/cmd 下发，字段按需求规格 ④；运动由控制台执行，本页不直接碰硬件。

import { MODES, MODE_NAMES, JOINT_LIMITS, NAMED_POSES } from "./contract.js";
import { sendCmd } from "./api.js";
import { toast } from "./format.js";
import { createSceneView } from "./scene3d.js";

const PRESETS = [["dance", "跳舞"], ["stack", "叠积木"], ["carry", "搬运"]];

async function cmd(payload, okMessage) {
  const started = performance.now();
  try {
    await sendCmd(payload.cmd, payload);
    toast(`${okMessage}（${Math.round(performance.now() - started)} ms）`, "good");
    return true;
  } catch (err) {
    toast(err.message, "bad");
    return false;
  }
}

export function createControlView(root) {
  root.innerHTML = `
    <section class="panel">
      <h2>操作权与夹爪 <small>控制台每次只认一个持有者；使能只有 2 s，多步运动期间由服务端续期</small></h2>
      <div class="row">
        <button id="enableBtn" class="primary">使能</button>
        <button id="disarmBtn" class="danger">解除</button>
        <span class="chip" id="armChip">—</span>
        <span class="sep"></span>
        <button id="stopBtn" class="danger">停止</button>
      </div>
      <div class="row">
        <span class="dim">夹爪</span>
        <button data-grip="grip">合拢</button>
        <button data-grip="open">张开</button>
        <button data-grip="wide">张到最大</button>
        <span class="dim" id="gripState">—</span>
      </div>
      <p class="hint" id="opHint"></p>
    </section>
    <section class="panel">
      <h2>作业模式 <small>W1 · 手动 / 预设动作 / 视觉自主，切换响应 ≤ 1 s</small></h2>
      <div class="row" id="modeRow"></div>
      <p class="hint" id="modeHint">当前模式由服务端持有，切换会写入 mode_changed 事件。</p>
    </section>
    <div class="grid-2">
      <section class="panel">
        <h2>六轴角度 <small>W2 · 拖动滑块后点“下发”，统一走六舵机批量写入</small></h2>
        <div id="jointBars" class="joints"></div>
        <p class="hint" id="jointHint"></p>
        <div class="row">
          <button id="sendJoints" class="primary">下发角度</button>
          <button id="poseUpright">直立</button>
          <button id="poseObserve">观察位</button>
        </div>
      </section>
      <section class="panel">
        <h2>三维状态视图 <small>骨架点取自服务端正运动学回读，拖动旋转、滚轮缩放</small></h2>
        <canvas id="armCanvas" class="arm"></canvas>
        <h3>坐标移动 <small>基座坐标系，厘米</small></h3>
        <div class="row">
          <label>X <input id="tx" type="number" step="0.5" value="0"></label>
          <label>Y <input id="ty" type="number" step="0.5" value="16"></label>
          <label>Z <input id="tz" type="number" step="0.5" value="2"></label>
          <button id="moveTo">移动到</button>
        </div>
        <h3>一键任务 <small>W2 · 动作由硬件组提供</small></h3>
        <div class="row" id="presetRow"></div>
        <p class="hint" id="presetHint">跳舞、叠积木、搬运这三个动作本身由硬件组（H2-8）提供；接进来之前点下去会明确返回“尚未接入”，而不是假执行。</p>
      </section>
    </div>`;

  const $ = (id) => root.querySelector("#" + id);
  const modeRow = $("modeRow");
  modeRow.innerHTML = MODES.map((m) =>
    `<button data-mode="${m}" class="mode">${MODE_NAMES[m]}</button>`).join("");
  modeRow.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-mode]");
    if (btn) cmd({ cmd: "set_mode", mode: btn.dataset.mode }, `已切到${MODE_NAMES[btn.dataset.mode]}`);
  });

  const presetRow = $("presetRow");
  presetRow.innerHTML = PRESETS.map(([p, label]) =>
    `<button data-preset="${p}">${label}</button>`).join("");
  presetRow.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-preset]");
    if (btn) cmd({ cmd: "run_preset", preset: btn.dataset.preset }, `已下发“${btn.textContent}”`);
  });

  const bars = $("jointBars");
  bars.innerHTML = JOINT_LIMITS.map((j, i) => `
    <div class="joint" data-i="${i}">
      <span>${j.name}</span>
      <strong>—</strong>
      <input type="range" min="${j.min}" max="${j.max}" step="1" value="${Math.round((j.min + j.max) / 2)}">
      <em class="dim">回读 —</em>
    </div>`).join("");

  const readJoint = (i) => +bars.querySelector(`.joint[data-i="${i}"] input`).value;
  const setJoint = (i, v) => {
    const input = bars.querySelector(`.joint[data-i="${i}"] input`);
    if (v == null) return;
    input.value = Math.round(v);
    input.closest(".joint").querySelector("strong").textContent = `${Math.round(v)}°`;
  };

  // 拖动时立刻把目标角度显示出来，不等 0.4 s 的状态轮询——之前只能等机械臂动起来才有数
  bars.addEventListener("input", (ev) => {
    const el = ev.target.closest(".joint");
    if (el) el.querySelector("strong").textContent = `${Math.round(+ev.target.value)}°`;
  });

  $("sendJoints").addEventListener("click", () =>
    cmd({ cmd: "set_joints", joints_deg: JOINT_LIMITS.map((_, i) => readJoint(i)) }, "角度已下发"));
  $("poseUpright").addEventListener("click", () => applyPose(NAMED_POSES.upright));
  $("poseObserve").addEventListener("click", () => applyPose(NAMED_POSES.observe));
  $("stopBtn").addEventListener("click", () => cmd({ cmd: "stop" }, "已停止"));
  $("enableBtn").addEventListener("click", () => cmd({ cmd: "enable" }, "已取得操作权"));
  $("disarmBtn").addEventListener("click", () => cmd({ cmd: "stop" }, "已解除操作权"));
  root.querySelectorAll("button[data-grip]").forEach((btn) =>
    btn.addEventListener("click", () =>
      cmd({ cmd: "gripper", action: btn.dataset.grip }, `夹爪“${btn.textContent}”已下发`)));
  $("moveTo").addEventListener("click", () => cmd({
    cmd: "move_to",
    target_cm: [+$("tx").value, +$("ty").value, +$("tz").value],
  }, "移动指令已下发"));

  function applyPose(pose) {
    pose.forEach((v, i) => { if (v != null) setJoint(i, v); });
    toast("已填入预设姿态，点“下发角度”执行");
  }

  const scene = createSceneView($("armCanvas"));

  return {
    update(store) {
      const mode = store.status.mode ?? "manual";
      modeRow.querySelectorAll("button").forEach((b) =>
        b.classList.toggle("on", b.dataset.mode === mode));

      const arm = store.status.arm ?? {};
      const chip = $("armChip");
      chip.textContent = !arm.attached ? "未接控制台" : arm.armed ? "已使能" : "未使能";
      chip.className = "chip " + (arm.armed ? "on" : "warn");
      $("enableBtn").disabled = !arm.attached || !!arm.armed || !!arm.busy;
      $("disarmBtn").disabled = !arm.armed;
      $("gripState").textContent = !arm.attached ? "—"
        : `${arm.grip_action ?? "—"} · 回读 ${arm.grip_readback ?? "—"}° · ` +
          `${arm.holding ? "夹住" : arm.grip_result === "empty" ? "夹空" : arm.grip_result ?? "—"}`;
      $("opHint").textContent = !arm.attached
        ? "未接控制台，运动类指令会返回 501。启动时加 --console http://127.0.0.1:8770 接仿真，或指向实机控制台。"
        : arm.error ? `控制台错误：${arm.error}`
        : arm.armed ? "已持有操作权，服务端每 1 s 续期。"
        : arm.busy ? "正在运动中，等这一步结束。" : "先点“使能”取得操作权，再下发角度或坐标。";

      // 控制台给的是它自己的软限位（core.limits），收到就以它为准：滑块不再给出送不出去的角度
      const limitOf = (i) => {
        const l = arm.limits?.[String(i + 1)];
        return l ? { min: l[0], max: l[1] } : JOINT_LIMITS[i];
      };

      const pose = store.status.joints_deg ?? store.lastPose;
      bars.querySelectorAll(".joint").forEach((el) => {
        const i = +el.dataset.i, lim = limitOf(i);
        const input = el.querySelector("input");
        if (+input.min !== lim.min) input.min = lim.min;
        if (+input.max !== lim.max) input.max = lim.max;
        // 限位变了之后滑块值可能落在界外，先夹回来再显示
        const clamped = Math.min(lim.max, Math.max(lim.min, +input.value));
        if (+input.value !== clamped) input.value = clamped;
        // strong = 滑块目标（拖动实时变），em = 执行侧回读（0.4 s 一拍）
        el.querySelector("strong").textContent = `${Math.round(clamped)}°`;
        const back = pose?.[i];
        el.querySelector("em").textContent = `回读 ${back == null ? "—" : Math.round(back)}°`;
        el.querySelector("em").className = back != null && (back < lim.min || back > lim.max) ? "bad" : "dim";
      });

      // 指令不排队：上一段没走完时下发会被拒，先把按钮按住并说明原因
      $("sendJoints").disabled = !!arm.busy;
      $("sendJoints").textContent = arm.busy ? "运动中…" : "下发角度";

      // 限位档与两个容易误会的行为，写明白省得当故障报
      $("jointHint").textContent = !arm.attached
        ? "未接控制台，滑块用默认限位；接上后以控制台当前档位为准。"
        : (arm.profile === "explore"
          ? "限位档：探索——滑块范围即课程给的舵机范围（J2–J4 可到 −10°，J5 90–270°）。注意实机上 J5 超过 180° 会挡住腕部相机；标准档会把各关节再收 5° 余量。"
          : `限位档：${arm.profile ?? "标准"}——课程范围向内收 5° 安全余量（J1/J5 因走线短收得更多）。`)
          + " J6 是夹爪不是普通关节：空载完全闭合的读数约 132°，指令再大也会触发阻力保护停在原地（实机相同），不是故障。"
          + " 运动按每段 ≤15° 分段执行、六轴同段同动，段间回读校验、遇阻力立即停——所以大角度要几秒，且一次只执行一条。";
      scene.set(arm.chain_cm ?? null);
    },
  };
}

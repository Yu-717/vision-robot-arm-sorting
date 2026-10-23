// 装配：页签、事件流、状态推送。页面只依赖日志契约，实机与仿真共用同一套路径。

import { Store } from "./store.js";
import { LogFeed, StatusFeed, fetchLog } from "./api.js";
import { createLiveView } from "./live.js";
import { createFaultsView } from "./faults.js";
import { createControlView } from "./control.js";
import { createTimersView } from "./timers.js";
import { createUnknownView } from "./unknown.js";
import { MODE_NAMES, CONTRACT_VERSION } from "./contract.js";
import { toast } from "./format.js";

const TABS = [
  ["live", "实时状态"], ["faults", "异常日志"], ["control", "控制面板"],
  ["timers", "定时任务"], ["unknown", "未知物体"],
];
const BACKFILL = 2000;                  // 首屏最多回放的历史行数

const store = new Store();
const root = document.getElementById("views");
const nav = document.getElementById("tabs");

const views = {};
for (const [key, label] of TABS) {
  const section = document.createElement("section");
  section.className = "view";
  section.hidden = true;
  root.appendChild(section);
  views[key] = { section, label, view: null };
}
views.live.view = createLiveView(views.live.section);
views.faults.view = createFaultsView(views.faults.section);
views.control.view = createControlView(views.control.section);
views.timers.view = createTimersView(views.timers.section);
views.unknown.view = createUnknownView(views.unknown.section);

let active = "live";
nav.innerHTML = TABS.map(([key, label]) =>
  `<button data-tab="${key}"${key === active ? ' class="on"' : ""}>${label}</button>`).join("");
nav.addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-tab]");
  if (!btn) return;
  active = btn.dataset.tab;
  nav.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
  render();
});

const chip = (id) => document.getElementById(id);

function render() {
  for (const [key, v] of Object.entries(views)) v.section.hidden = key !== active;
  views[active].view?.update(store);

  chip("chipConn").textContent = { connecting: "连接中", live: "实时", polling: "轮询", error: "重连中" }[store.connection] ?? "—";
  chip("chipConn").className = "chip " + (store.connection === "live" ? "on" : store.connection === "error" ? "warn" : "");
  chip("chipMode").textContent = `模式 ${MODE_NAMES[store.status.mode] ?? store.status.mode ?? "—"}`;
  chip("chipLatency").textContent = store.latency === null ? "延迟 —" : `延迟 ${store.latency.toFixed(2)} s`;
  chip("chipTotal").textContent = `${store.events.length} 事件`;
}

store.subscribe(render);

// 首屏把已有日志回放一遍，再接实时流：条数一致按这个核对
(async () => {
  try {
    const { events } = await fetchLog(0, BACKFILL);
    events.forEach((e) => store.push(e));
  } catch { /* 服务未起时页面仍可用 */ }
  await store.refresh();
  render();

  const feed = new LogFeed({
    onEvent: (e) => {
      store.push(e);
      store.latency = feed.latency;
      if (store.events.length % 20 === 0) store.refresh();
    },
    onState: (s) => { store.connection = s; render(); },
  });
  feed.start(0);

  new StatusFeed({
    onStatus: (s) => store.setStatus(s),
  }).start();

  setInterval(() => store.refresh(), 5000);
  setInterval(() => { store.latency = feed.latency; render(); }, 1000);
})();

document.getElementById("contract").textContent = `日志契约 v${CONTRACT_VERSION}`;

// 控制台：运动指令转发给它（仿真或实机），它自己的 3D 工作台也顺便链过去。
// 那个页面内部还用 /api/sim/* 之类的路径，硬代理过来会半残，所以只做外链不做内嵌。
fetch("/api/health").then((r) => r.json()).then((h) => {
  const chip = document.getElementById("chipArm");
  chip.textContent = h.console ? `控制台 ${h.console.replace(/^https?:\/\//, "")}` : "控制台 未接";
  chip.className = "chip " + (h.console_ready ? "on" : "warn");
  const link = document.getElementById("simLink");
  if (h.console) {
    link.href = h.console + "/sim/";
    link.hidden = false;
    link.title = h.console_ready ? "执行侧的 3D 机械臂与合成相机画面" : "控制台还没就绪";
  }
}).catch(() => { });
window.addEventListener("resize", render);
window.addEventListener("error", (e) => toast(e.message, "bad"));

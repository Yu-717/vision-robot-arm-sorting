// 页面共享的派生状态：从事件流推出当前轮次、计数、模式与角度。
// 轮次靠事件先后顺序推断（日志契约 §4：逐块事件里没有轮次编号）。

import { fetchRecords, fetchSummary } from "./api.js";
import { EVENTS } from "./contract.js";

const EMPTY_COUNTS = () => ({ sorted: 0, missed: 0, skipped: 0, failed: 0, cleared: 0 });

export class Store {
  constructor() {
    this.events = [];            // 契约内事件，按 seq 升序
    this.records = [];           // 逐块记录（六项）
    this.summary = null;
    this.status = { mode: "manual", joints_deg: null, t: null };
    this.rounds = [];            // 已结束的轮
    this.current = null;         // 进行中的轮（可能一直没有结束，见契约 §7）
    this.connection = "connecting";
    this.latency = null;
    this._roundNo = 0;
    this._listeners = new Set();
  }

  subscribe(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
  notify() { for (const fn of this._listeners) fn(this); }

  push(event) {
    this.events.push(event);
    this._track(event);
    this.notify();
  }

  _track(e) {
    const name = e.event;
    if (name === "pipeline_start") {
      this._roundNo += 1;
      this.current = { no: this._roundNo, run: e.run, t: e.t, cubes: e.cubes?.length ?? 0,
                       seconds: null, ...EMPTY_COUNTS() };
      this.rounds = [...this.rounds];
    } else if (name === "pipeline_end") {
      if (this.current) {
        Object.assign(this.current, { seconds: e.seconds, sorted: e.sorted, missed: e.missed,
                                      skipped: e.skipped, cleared: e.cleared, ended: e.t });
        this.rounds = [...this.rounds, this.current];
      }
      this.current = null;
    } else if (name === "mode_changed") {
      this.status = { ...this.status, mode: e.mode };
    }
    const result = EVENTS[name]?.record;
    if (result && this.current) this.current[result] += 1;
  }

  setStatus(status) {
    this.status = { ...this.status, ...status };
    this.notify();
  }

  /** 记录与统计不随每次事件刷新，按需拉（1 000 条以内很快） */
  async refresh() {
    try {
      const [records, summary] = await Promise.all([fetchRecords(), fetchSummary()]);
      this.records = records.records ?? [];
      this.summary = summary;
    } catch { /* 保持上一次 */ }
    this.notify();
  }

  /** 最近一个带 pose 的事件，用来显示角度与“在做什么” */
  get lastPose() {
    for (let i = this.events.length - 1; i >= 0; i--) {
      const pose = this.events[i].pose;
      if (Array.isArray(pose) && pose.length === 6) return pose;
    }
    return null;
  }

  get lastEvent() { return this.events[this.events.length - 1] ?? null; }

  /** 已处理块数（本轮） */
  get currentHandled() {
    if (!this.current) return 0;
    const c = this.current;
    return c.sorted + c.missed + c.skipped + c.failed + c.cleared;
  }
}

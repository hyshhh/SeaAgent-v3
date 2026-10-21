// 极简 DOM 桩：让 draw-archive-flow.mjs 能在 Node 中直接执行。
// 度量数据由真实浏览器注入到 globalThis.__MEA__(text, size, weight, family, spacing)。
const noop = () => {};
const makeEl = () => ({
  setAttribute: noop,
  appendChild: noop,
  _t: "",
  set textContent(v) { this._t = v; },
  get textContent() { return this._t; },
  getComputedTextLength() {
    const f = globalThis.__MEA__;
    return f ? f(this._t || "", globalThis.__MEA_SIZE__ || 28, globalThis.__MEA_WEIGHT__ || "400", globalThis.__MEA_FAMILY__ || "", 0) : 0;
  }
});
globalThis.document = { createElementNS: makeEl, createElement: makeEl, body: { appendChild: noop } };
globalThis.window = globalThis;
/**
 * capture-png.mjs —— 用 Chrome DevTools Protocol 把 SVG 精确渲染成 PNG。
 *
 * 为什么不用 Edge 的 --screenshot：无头模式对截图尺寸有约 750px 的硬上限。
 * 为什么不用 print-to-pdf：Chrome 打印会把粗体中文回退成 Type 3 (Noto Serif SC) 字体子集，
 *   pdftoppm/多数 PDF 渲染器画不出来（标题会整块消失）。
 * 走 CDP：Emulation.setDeviceMetricsOverride 指定精确像素 + Page.captureScreenshot，
 *   渲染结果与屏幕上所见完全一致。
 *
 * 用法： node capture-png.mjs <edge路径> <svg绝对路径> <输出png> <宽> <高> [缩放]
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [, , EDGE, SVG, OUT, W, H, SCALE] = process.argv;
const width = parseInt(W, 10);
const height = parseInt(H, 10);
const scale = SCALE ? parseFloat(SCALE) : 2;
const port = 9222 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "dsh-edge-"));

const child = spawn(EDGE, [
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--hide-scrollbars",
  "--force-color-profile=srgb",
  "--disable-extensions",
  "--remote-debugging-port=" + port,
  "--user-data-dir=" + profile,
  "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function wsUrl() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch("http://127.0.0.1:" + port + "/json/version");
      const j = await r.json();
      if (j.webSocketDebuggerUrl) return j.webSocketDebuggerUrl;
    } catch { /* 还没起来 */ }
    await sleep(250);
  }
  throw new Error("无法连接 Edge 调试端口");
}

const ws = new WebSocket(await wsUrl());
await new Promise((res, rej) => { ws.addEventListener("open", res); ws.addEventListener("error", rej); });

let seq = 0;
const pending = new Map();
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result);
  }
});
const send = (method, params, sessionId) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params: params || {}, sessionId }));
});

const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
await send("Page.enable", {}, sessionId);
await send("Emulation.setDeviceMetricsOverride",
  { width, height, deviceScaleFactor: scale, mobile: false }, sessionId);
await send("Page.navigate", { url: pathToFileURL(SVG).href }, sessionId);
await sleep(2500); // 等字体与 SVG 布局稳定
const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }, sessionId);
fs.writeFileSync(OUT, Buffer.from(shot.data, "base64"));

try { await send("Browser.close", {}); } catch { /* ignore */ }
try { ws.close(); } catch { /* ignore */ }
setTimeout(() => { try { child.kill(); } catch { /* ignore */ } ; process.exit(0); }, 400);
console.log("[ok] " + OUT + "  " + width * scale + " x " + height * scale);
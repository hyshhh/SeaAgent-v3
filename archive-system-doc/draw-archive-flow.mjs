/**
 * 档案库管理系统 —— 业务流程框图（泳道式）
 *
 * 三条泳道：读者/利用者 · 档案管理员 · 系统与管理员
 * 输出：archive-flow.svg（矢量，可进 Word/LaTeX/Illustrator 继续改）
 *      archive-flow.png（高清位图，由 render-png.ps1 调 Edge 无头渲染生成）
 *
 * 依赖：仅 Node.js（无第三方包）。文字宽度用浏览器原生度量，中文排版精确。
 */
import fs from "node:fs";
import path from "node:path";
import "./_svgdom.mjs";

const OUT = import.meta.dirname;
const SVG_FILE = path.join(OUT, "archive-flow.svg");

/* ------------------------------------------------------------------ 画布几何 */
const M = 48;                 // 画布外边距
const BLEED = 30;             // 打印安全边距（避免浏览器打印裁掉边缘）
const PB = 48;                // 泳道内边距
const GAP = 26;               // 节点间距
const LANE_GAP = 46;          // 泳道间距
const SIDE_W = 300;           // 旁支小框宽度
const SIDE_DX = 40;           // 旁支水平偏移
const F_TITLE = 52, F_HEAD = 40, F_SUB = 26, F_BODY = 28, F_SMALL = 25, F_FOOT = 26;

const FONT = "'Microsoft YaHei','Segoe UI','Times New Roman',serif";
const SANS = "'Microsoft YaHei','Segoe UI','Times New Roman',serif";

/* ------------------------------------------------------------------ 文字度量 */
let MEA = null;
function installMeasure() {
  const b = 1600, h = 400;
  const root = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  root.setAttribute("viewBox", "0 0 " + b + " " + h);
  root.setAttribute("width", b);
  root.setAttribute("height", h);
  root.setAttribute("style", "position:absolute;left:-4000px;top:0");
  const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
  el.setAttribute("x", "10");
  el.setAttribute("y", "200");
  root.appendChild(el);
  document.body.appendChild(root);
  MEA = (s, size, weight, family, spacing) => {
    el.setAttribute("font-size", size);
    el.setAttribute("font-weight", weight || "400");
    el.setAttribute("font-family", family || FONT);
    el.setAttribute("letter-spacing", spacing || 0);
    globalThis.__MEA_SIZE__ = size;
    globalThis.__MEA_WEIGHT__ = weight || "400";
    globalThis.__MEA_FAMILY__ = family || FONT;
    el.textContent = s;
    return el.getComputedTextLength();
  };
}

/* ------------------------------------------------------------------ 语义换行 */
const NO_HEAD = "，。、；：）】》”！？%·…";
const NO_TAIL = "（【《“";
function tokenize(s) {
  const out = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (/[0-9]/.test(c)) { let j = i; while (j < s.length && /[0-9.]/.test(s[j])) j++; out.push(s.slice(i, j)); i = j; }
    else if (/[A-Za-z]/.test(c)) { let j = i; while (j < s.length && /[A-Za-z0-9\-]/.test(s[j])) j++; out.push(s.slice(i, j)); i = j; }
    else { out.push(c); i++; }
  }
  return out;
}
function wrapText(text, maxW, size, weight, family) {
  const lines = [];
  for (const para of String(text).split("\n")) {
    const toks = tokenize(para);
    let line = "";
    for (const t of toks) {
      const cand = line + t;
      if (line && MEA(cand, size, weight, family) > maxW) {
        const head = line[line.length - 1], tail = t[0];
        if (NO_HEAD.includes(tail) || NO_TAIL.includes(head)) { lines.push(cand); line = ""; }
        else { lines.push(line); line = t; }
      } else line = cand;
    }
    lines.push(line);
  }
  return lines;
}

/* ------------------------------------------------------------------ SVG 构件 */
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const n = (v) => Math.round(v * 100) / 100;

function roundRect(x, y, w, h, fill, stroke, o) {
  o = o || {};
  const r = o.r === undefined ? 16 : o.r, lw = o.lw === undefined ? 2 : o.lw;
  let s = "";
  if (o.sw) s += '<rect x="' + n(x) + '" y="' + n(y + 4) + '" width="' + n(w) + '" height="' + n(h) + '" rx="' + r + '" fill="#0b2540" opacity="' + o.sw + '"/>';
  s += '<rect x="' + n(x) + '" y="' + n(y) + '" width="' + n(w) + '" height="' + n(h) + '" rx="' + r + '" fill="' + fill + '" stroke="' + stroke + '" stroke-width="' + lw + '"';
  if (o.dash) s += ' stroke-dasharray="' + o.dash + '"';
  return s + "/>";
}
function diamond(cx, cy, w, h, fill, stroke, lw) {
  const p = n(cx) + "," + n(cy - h / 2) + " " + n(cx + w / 2) + "," + n(cy) + " " + n(cx) + "," + n(cy + h / 2) + " " + n(cx - w / 2) + "," + n(cy);
  return '<polygon points="' + p + '" fill="' + fill + '" stroke="' + stroke + '" stroke-width="' + (lw || 2.6) + '"/>';
}
function arrow(x1, y1, x2, y2, color, lw, marker) {
  return '<line x1="' + n(x1) + '" y1="' + n(y1) + '" x2="' + n(x2) + '" y2="' + n(y2) + '" stroke="' + color +
    '" stroke-width="' + (lw || 2.6) + '" marker-end="url(#' + (marker || "arrow") + ')"/>';
}
function elbow(pts, color, o) {
  o = o || {};
  const d = pts.map((p, i) => (i ? "L" : "M") + n(p[0]) + "," + n(p[1])).join(" ");
  return '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="' + (o.lw || 2.2) +
    '" stroke-linejoin="round"' + (o.dash ? ' stroke-dasharray="' + o.dash + '"' : "") + ' marker-end="url(#arrow)"/>';
}
function textBlock(lines, cx, yTop, lh, o) {
  o = o || {};
  const size = o.size || F_BODY, weight = o.weight || "400", family = o.family || FONT, color = o.color || "#12222e";
  return lines.map((t, i) =>
    '<text x="' + n(cx) + '" y="' + n(yTop + lh * (i + 0.78)) + '" font-size="' + size + '" font-weight="' + weight +
    '" font-family="' + family + '" fill="' + color + '" text-anchor="middle">' + esc(t) + "</text>").join("");
}
/* ------------------------------------------------------------------ 泳道内容数据 */
const READER = [
  { t: "注册 / 登录", d: "账号密码登录，或校内统一身份认证（读者证号 + 密码）。" },
  { t: "身份与权限校验", d: ["未通过：提示错误，累计失败 3 次锁定账号", "通过：按读者 / 档案员 / 管理员分配角色权限"] },
  { t: "进入档案门户", d: "首页：本馆概况、新入藏档案、通知公告、热门档案、办事指南。" },
  { t: "目录浏览", d: "按全宗—目录—案卷—文件四级目录逐层浏览馆藏档案。" },
  { t: "多条件检索", d: "题名 / 责任者 / 形成时间 / 主题词 / 档号 / 全文关键词组合检索。" },
  { t: "检索结果列表", d: ["分页返回命中条目，显示题名、时间、保管期限与状态；", "命中为空则提示扩大范围或改用模糊检索。"] },
  { t: "案卷详情与全文预览", d: "元数据 + 全文 / 影像预览 + 开放程度说明 + 借阅申请入口。" },
  { t: "权限判定", kind: "diamond", d: ["开放档案：直接在线阅读", "控制 / 涉密档案：需审批并留痕"] },
  { t: "阅览 / 下载", d: "在线阅览、打印、受控下载；系统自动记录访问日志。" },
  { t: "提交借阅申请", d: "填写用途与借期，选择原件借出或复制件，提交进入审批队列。" },
  { t: "借出登记", d: "审批通过后办理出库，登记借阅人与归还期限，生成借阅凭证。" },
  { t: "到期提醒", d: "到期前 3 天提醒；逾期未还暂停借阅权限并进入催还流程。" },
  { t: "归还 / 破损处理", d: "核验档案完整性与密级，放回库位更新状态；破损则登记异常并补复制件。" },
  { t: "退出系统", side: true, d: "会话销毁；超时自动登出。" }
];

const CLERK = [
  { t: "登录后台管理系统", d: "使用档案员账号，进入内网 B/S 管理端，功能按角色授权。" },
  { t: "归类分流", kind: "diamond", d: ["新收档案：进入著录流程", "借阅申请：进入审批流程"] },
  { t: "著录与标引", d: ["登记档号、题名、责任者、形成时间、保管期限、密级、载体形态；", "按分类方案归类并标引主题词。"] },
  { t: "OCR / 全文提取", d: "纸质扫描、电子文件解析，生成全文文本与检索索引。" },
  { t: "著录校验与查重", d: "档号唯一性、字段规范、重复件核查；不合格退回补录。" },
  { t: "实体上架与数字化归档", d: ["纸质原件：库房排架并登记库位号（密集架—柜—格）；", "电子档案：入库服务器双副本，做防篡改校验。"] },
  { t: "库位索引", d: "建立「档号 ↔ 库位 / RFID / 条码」映射，支持扫码快速定位。" },
  { t: "档案调取", d: "按档号或扫码调取原件与复制件，登记出库；核对库位并更新状态。" },
  { t: "借阅审批", kind: "diamond", d: ["一般档案：档案员直接审批", "重要 / 控制档案：报主管领导审批"] },
  { t: "归还核验与续借", d: "核对份数与完整性，办理续借；异常情况登记并上报。" },
  { t: "到期鉴定", kind: "diamond", d: ["已达保管期限：提请鉴定销毁", "未到期：继续保存"] },
  { t: "鉴定与销毁", d: "鉴定小组复审，报批后监销并登记销毁清册；未通过则继续保管。" },
  { t: "退出系统", side: true, d: "登出并保留操作日志。" }
];

const SYS = [
  { t: "系统自动运行", d: "定时任务：逾期提醒、库房盘点、日志归档、统计报表。" },
  { t: "用户 / 角色 / 权限维护", d: "账号增删改、角色授权、权限矩阵配置、密级分级管控。" },
  { t: "审计日志", d: "记录操作、访问与借阅行为，支持追溯到人、防篡改留存。" },
  { t: "数据备份与恢复", d: "全量 + 增量备份，异地 / 异质保存，定期恢复演练。" }
];

/* ------------------------------------------------------------------ 版面计算 */
installMeasure();

const laneDefs = [
  { key: "reader", title: "读者 / 利用者", sub: "档案查询 · 借阅利用", items: READER,
    fill: "#f4f8fd", edge: "#3f6fb5", headFrom: "#4d7fc4", headTo: "#35619f",
    nodeFill: "#e7f0fa", nodeEdge: "#5b8ac6", bodyW: 1150 },
  { key: "clerk", title: "档案管理员", sub: "收集 · 整理 · 保管 · 利用", items: CLERK,
    fill: "#f5faf5", edge: "#3f8a55", headFrom: "#4f9c65", headTo: "#357a4a",
    nodeFill: "#e8f4ea", nodeEdge: "#5c9d70", bodyW: 1150 },
  { key: "sys", title: "系统与管理员", sub: "权限 · 安全 · 备份 · 统计", items: SYS,
    fill: "#f8f6fa", edge: "#7b5ea8", headFrom: "#8a6cb8", headTo: "#6c5097",
    nodeFill: "#efe9f7", nodeEdge: "#9179bb", bodyW: SIDE_DX + SIDE_W }
];

const lanes = [];
{
  let x = M;
  for (const def of laneDefs) {
    const w = def.bodyW + 2 * PB;
    lanes.push(Object.assign({}, def, { x: x, w: w }));
    x += w + LANE_GAP;
  }
}
const totalW = lanes[lanes.length - 1].x + lanes[lanes.length - 1].w + M + 2 * BLEED;

const LH = 38, LH_S = 34;
for (const lane of lanes) {
  const innerW = lane.w - 2 * PB;
  lane.flowW = lane.items.some((it) => it.side) ? innerW - SIDE_DX - SIDE_W : innerW;
  lane.flowX = lane.x + PB;
  for (const it of lane.items) {
    const side = !!it.side;
    const w = side ? SIDE_W : lane.flowW;
    const maxW = w - 44;
    it._w = w;
    it._title = wrapText(it.t, maxW, side ? F_SUB : F_BODY, "700", SANS);
    it._desc = it.d ? wrapText(it.d, maxW, F_SMALL, "400", SANS) : [];
    const lh = side ? LH - 2 : LH;
    let h = 22 + it._title.length * lh + (it._desc.length ? 14 + it._desc.length * LH_S : 0) + 22;
    if (it.kind === "diamond") h = Math.max(h + 34, 176);
    it._h = Math.max(h, side ? 96 : 104);
  }
  lane._bodyH = lane.items.reduce((a, it) => a + it._h, 0) + GAP * (lane.items.length - 1);
  lane._bodyH = Math.max(lane._bodyH, 300);
}

const HEADER_H = 128;
const TOP = M + 22;
const OFFSET = 96 + 6;
const laneH = HEADER_H + 18 + OFFSET + Math.max.apply(null, lanes.map((l) => l._bodyH)) + 40;
const bodyTop = TOP + HEADER_H + 18 + OFFSET;
const VH = TOP + laneH + 46 + 46 + 30 + 2 * BLEED;

/* ------------------------------------------------------------------ 输出 SVG */
const defs = [
  "<defs>",
  '  <marker id="arrow" viewBox="0 0 12 12" refX="9.5" refY="6" markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">',
  '    <path d="M1,1.2 L11,6 L1,10.8 z" fill="#33506b"/></marker>',
  "<defs>".replace("<defs>", ""),
  lanes.map((l) => '  <linearGradient id="g_' + l.key + '" x1="0" y1="0" x2="0" y2="1">' +
    '<stop offset="0" stop-color="' + l.headFrom + '"/><stop offset="1" stop-color="' + l.headTo + '"/></linearGradient>').join("\n"),
  "</defs>"
].join("\n");

let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + n(totalW) + " " + n(VH) + '" width="' + n(totalW) +
  '" height="' + n(VH) + '" font-family="' + FONT + '">';
svg += defs;
svg += '<rect width="100%" height="100%" fill="#ffffff"/>';
svg += '<g transform="translate(' + BLEED + ',' + BLEED + ')">';
svg += '<rect x="0" y="0" width="' + n(totalW) + '" height="' + n(VH) + '" fill="none" stroke="#c9d6e2" stroke-width="3"/>';

/* 标题区 */
svg += '<rect x="' + M + '" y="' + TOP + '" width="' + n(totalW - 2 * M) + '" height="96" rx="20" fill="#f2f6fb" stroke="#3f6fb5" stroke-width="2.4"/>';
svg += '<text x="' + n(totalW / 2) + '" y="' + n(TOP + 44) + '" font-size="' + F_TITLE + '" font-weight="700" font-family="' + SANS +
  '" fill="#1c3f66" text-anchor="middle">档案库管理系统 业务流程框图</text>';
svg += '<text x="' + n(totalW / 2) + '" y="' + n(TOP + 82) + '" font-size="' + F_SUB + '" font-family="' + SANS +
  '" fill="#5b7893" text-anchor="middle">泳道 = 角色分工 · 实线 = 主干流程 · 虚线 = 泳道间协作 · 菱形 = 判断分支</text>';

/* 泳道底板 + 表头 */
for (const lane of lanes) {
  const y = TOP + HEADER_H + 18;
  const h = laneH - HEADER_H - 18;
  lane._y = y; lane._h = h;
  svg += roundRect(lane.x, y, lane.w, h, lane.fill, lane.edge, { r: 24, lw: 2.4 });
  const hx = lane.x + 14, hw = lane.w - 28, hy = TOP + 14;
  svg += '<rect x="' + n(hx) + '" y="' + n(hy) + '" width="' + n(hw) + '" height="' + HEADER_H + '" rx="18" fill="url(#g_' + lane.key +
    ')" stroke="' + lane.headTo + '" stroke-width="2"/>';
  svg += '<text x="' + n(hx + hw / 2) + '" y="' + n(hy + 58) + '" font-size="' + F_HEAD + '" font-weight="700" font-family="' + SANS +
    '" fill="#ffffff" text-anchor="middle">' + esc(lane.title) + "</text>";
  svg += '<text x="' + n(hx + hw / 2) + '" y="' + n(hy + 100) + '" font-size="' + F_SUB + '" font-family="' + SANS +
    '" fill="#e9f1fa" text-anchor="middle">' + esc(lane.sub) + "</text>";
}

/* 节点 */
for (const lane of lanes) {
  let cy = bodyTop + 16;
  const cxFlow = lane.flowX + lane.flowW / 2;
  for (const it of lane.items) {
    const side = !!it.side;
    const x = side ? lane.flowX + SIDE_DX : lane.flowX;
    const cx = side ? x + SIDE_W / 2 : cxFlow;
    it._x = x; it._y = cy; it._cx = cx; it._cy = cy + it._h / 2;
    const isDia = it.kind === "diamond";
    if (isDia) svg += diamond(cx, it._cy, lane.flowW - 40, it._h, "#fdf3df", "#c1783a", 2.6);
    else svg += roundRect(x, cy, it._w, it._h, lane.nodeFill, lane.nodeEdge, { r: 16, lw: 2.2, sw: 0.05 });
    const lh = side ? LH - 2 : LH;
    const titleTop = cy + (isDia ? 30 : 22);
    svg += textBlock(it._title, cx, titleTop, lh, { size: side ? F_SUB : F_BODY, weight: "700", family: SANS, color: "#12324f" });
    if (it._desc.length) {
      svg += textBlock(it._desc, cx, titleTop + it._title.length * lh + 12, LH_S, { size: F_SMALL, family: SANS, color: "#3d5568" });
    }
    cy += it._h + GAP;
  }
}

/* 主干箭头 + 旁支 */
for (const lane of lanes) {
  const cx = lane.flowX + lane.flowW / 2;
  for (let i = 0; i < lane.items.length - 1; i++) {
    const a = lane.items[i], b = lane.items[i + 1];
    if (a.side || b.side) continue;
    svg += arrow(cx, a._y + a._h, cx, b._y - 2, lane.nodeEdge, 2.6);
  }
  for (let i = 1; i < lane.items.length; i++) {
    const it = lane.items[i];
    if (!it.side) continue;
    const prev = lane.items[i - 1];
    svg += elbow([[cx, prev._y + prev._h], [cx, it._cy], [it._x - 8, it._cy]], lane.nodeEdge, { lw: 2.2 });
  }
}

/* 顶部统一入口 */
{
  const cx = totalW / 2;
  const y0 = TOP + 96;
  const y1 = bodyTop - 44;
  svg += '<rect x="' + n(cx - 470) + '" y="' + n(y0) + '" width="940" height="' + (OFFSET - 6) + '" rx="30" fill="#1f4a78"/>';
  svg += '<text x="' + n(cx) + '" y="' + n(y0 + 56) + '" font-size="34" font-weight="700" font-family="' + SANS +
    '" fill="#ffffff" text-anchor="middle">用户 / 管理员 访问入口</text>';
  for (const lane of lanes) {
    const lx = lane.flowX + lane.flowW / 2;
    svg += elbow([[cx, y0 + OFFSET - 6], [cx, y1 + 24], [lx, y1 + 24], [lx, lane.items[0]._y - 2]], lane.edge, { lw: 2.8 });
  }
}

/* 泳道间协作箭头（虚线） */
function crossLink(aLane, aIdx, bLane, bIdx, label, opt) {
  opt = opt || {};
  const L = lanes[aLane], R = lanes[bLane];
  const a = L.items[aIdx], b = R.items[bIdx];
  const x1 = L.flowX + L.flowW + 4, x2 = R.flowX - 6;
  const y = (opt.y !== undefined) ? opt.y : (a._cy + b._cy) / 2;
  const d = "M" + n(x1) + "," + n(a._cy) + " H" + n(x1 + 22) + " V" + n(y) + " H" + n(x2 - 22) + " V" + n(b._cy) + " H" + n(x2) + " H" + n(R.flowX - 2);
  let s = '<path d="' + d + '" fill="none" stroke="#8a9bad" stroke-width="2.2" stroke-linejoin="round" stroke-dasharray="12 8" marker-end="url(#arrow)"/>';
  s += '<text x="' + n((x1 + 22 + x2 - 22) / 2) + '" y="' + n(y - 12) + '" font-size="' + (F_SUB - 2) + '" font-family="' + SANS +
    '" fill="#6b7f92" text-anchor="middle">' + esc(label) + "</text>";
  return s;
}
svg += crossLink(0, 9, 1, 8, "提交借阅申请");
svg += crossLink(1, 11, 2, 2, "销毁清册 / 鉴定记录归档", { y: lanes[2].items[2]._y - 34 });

/* 旁支虚线：读者身份信息 → 系统权限数据 */
{
  const L = lanes[0], R = lanes[2];
  const a = L.items[1], b = R.items[1];
  const y = a._cy + 40;
  const x1 = L.flowX + L.flowW + 4, x2 = R.flowX - 6;
  const d = "M" + n(x1) + "," + n(a._cy) + " H" + n(x1 + 14) + " V" + n(y) + " H" + n(x2 - 14) + " V" + n(b._cy) + " H" + n(x2);
  svg += '<path d="' + d + '" fill="none" stroke="#a9b7c6" stroke-width="1.8" stroke-dasharray="4 7" marker-end="url(#arrow)"/>';
  svg += '<text x="' + n((x1 + x2) / 2) + '" y="' + n(y - 12) + '" font-size="' + F_SUB + '" font-family="' + SANS +
    '" fill="#7b8ea3" text-anchor="middle">读者身份信息 → 权限数据</text>';
}

/* 判断分支注记 */
{
  const lane = lanes[0];
  const it = lane.items[7];
  const cx = it._cx;
  svg += '<path d="M' + n(cx + (lane.flowW - 40) / 2 - 4) + "," + n(it._cy) + " H" + n(cx + lane.flowW / 2 - 30) +
    ' V' + n(it._cy - 46) + ' H' + n(cx + 60) + '" fill="none" stroke="#c1783a" stroke-width="1.8" stroke-dasharray="8 6" marker-end="url(#arrow)"/>';
  svg += '<text x="' + n(cx + lane.flowW / 2 + 16) + '" y="' + n(it._cy - 8) + '" font-size="' + F_SMALL + '" font-family="' + SANS +
    '" fill="#a8662f">未授权 → 返回浏览</text>';
}

/* 图注 */
{
  const y = TOP + laneH + 34;
  const items = [
    ["▢", "#5b8ac6", "读者操作"],
    ["▢", "#5c9d70", "档案管理员操作"],
    ["▢", "#9179bb", "系统 / 管理员操作"],
    ["◇", "#c1783a", "判断分支"],
    ["┈▶", "#8a9bad", "泳道间协作 / 数据传递"]
  ];
  let x = M + 6;
  for (const pair of items) {
    const sym = pair[0], col = pair[1], label = pair[2];
    svg += '<text x="' + n(x) + '" y="' + n(y + 26) + '" font-size="30" font-weight="700" font-family="' + SANS + '" fill="' + col + '">' + sym + "</text>";
    const w = MEA(sym, 30, "700", SANS) + 12;
    svg += '<text x="' + n(x + w) + '" y="' + n(y + 26) + '" font-size="' + F_FOOT + '" font-family="' + SANS + '" fill="#41586e">' + label + "</text>";
    x += w + MEA(label, F_FOOT, "400", SANS) + 54;
  }
  svg += '<text x="' + n(totalW - M - 6) + '" y="' + n(y + 26) + '" font-size="' + F_FOOT + '" font-family="' + SANS +
    '" fill="#8a9bad" text-anchor="end">绘制：draw-archive-flow.mjs（Node.js + Edge 无头渲染）</text>';
}
svg += "</g>";
svg += "</svg>";

fs.writeFileSync(SVG_FILE, svg, "utf8");
/* 供浏览器度量使用：把全部文案与字符集交给 render-png.ps1 注入页面 */
{
  const texts = [];
  for (const def of laneDefs) for (const it of def.items) { texts.push(it.t); if (it.d) for (const d of it.d) texts.push(d); }
  const chars = Array.from(new Set(texts.join("").split(""))).sort().join("");
  fs.writeFileSync(path.join(OUT, "_strings.json"), JSON.stringify({ texts: texts, chars: chars }), "utf8");
}
fs.writeFileSync(path.join(OUT, "svg-size.json"), JSON.stringify({ width: Math.round(totalW), height: Math.round(VH) }), "utf8");
console.log("[ok] " + SVG_FILE + "  " + Math.round(totalW) + " x " + Math.round(VH));

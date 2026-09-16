"""SeaAgent 三子智能体架构图 v2（Planning / Execution / Reflection）。

设计依据（与代码一致）：
  · harness/middleware.py:66  TodoListMiddleware 挂在主智能体（= Execution Agent）中间件栈上，
    write_todos 注入给 Execution，由它维护 plan 的实时状态。
  · config/subagents.yaml     原 track_scout / registry_checker / visual_prover 三个从智能体
    收进 Execution Agent，成为它调用的领域工具，不再单独成框。
  · harness/wrapup.py         EvidenceWrapUpMiddleware 收尾把关；证据经 show_evidence
    走代码路径富化渲染到前端证据面板。

输出：PNG + SVG（可进 LaTeX / Illustrator 继续编辑）。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- 字体
def resolve(cands):
    from matplotlib.font_manager import FontProperties, findfont
    for name in cands:
        try:
            path = findfont(FontProperties(family=name), fallback_to_default=False)
        except Exception:
            continue
        if path:
            print(f"[font] {name} -> {path}")
            return name
    raise RuntimeError(f"none of {cands} resolved")

CJK = resolve(["Microsoft YaHei", "SimHei", "SimSun"])
LATIN = resolve(["Times New Roman", "Cambria", "Georgia", "DejaVu Serif"])

# ---------------------------------------------------------------- 画布
W, H = 52.0, 21.6
fig = plt.figure(figsize=(19.5, 8.1), dpi=110)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

C = {
    "plan": "#dbe7f6", "plan_e": "#3f6fb5",
    "exec": "#dcefe0", "exec_e": "#3f8a55",
    "refl": "#fbe6d2", "refl_e": "#c1783a",
    "mem":  "#f0f0f0", "mem_e":  "#6b6b6b",
    "tool": "#f6ecf6", "tool_e": "#8a5a8a",
    "io":   "#ffffff", "io_e":   "#333333",
    "accent": "#f3f8ff", "accent_e": "#5b7fbf",
    "arrow": "#333333",
}
GREY = "#5a5a5a"

def box(x, y, w, h, fc, ec, lw=1.3, r=0.22, ls="solid", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc,
                 linestyle=ls, zorder=z, mutation_aspect=1.0))

def txt(x, y, s, size=11, weight="normal", color="black", ha="center",
        va="center", family=None, style="normal", z=6):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            zorder=z, style=style, family=family or LATIN)

def bullet_list(x, y_top, items, size=9.6, dy=0.46, color="#222222"):
    for i, s in enumerate(items):
        ax.text(x, y_top - i * dy, s, fontsize=size, color=color,
                ha="left", va="center", zorder=6, family=CJK)

def arrow(p1, p2, color=None, lw=2.0, style="-|>", rad=0.0, ls="solid", z=4, ms=14):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                 linewidth=lw, color=color or C["arrow"], zorder=z, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

def label(x, y, s, size=9.0, color=GREY, pad=0.14):
    """带白色底衬的箭头标签，避免压在框线上看不清。"""
    ax.text(x, y, s, fontsize=size, color=color, ha="center", va="center",
            zorder=8, family=CJK,
            bbox=dict(boxstyle=f"round,pad={pad}", fc="white", ec="none", alpha=0.92))

# ---------------------------------------------------------------- 标题
txt(0.6, 21.05, "SeaAgent", size=19, weight="bold", style="italic", ha="left")
txt(6.0, 21.05, "Three Sub-Agent Coordination: Planning · Execution · Reflection",
    size=12.5, weight="bold", color=GREY, ha="left", family=CJK)
ax.plot([0.6, W - 0.6], [20.35, 20.35], color="#bbbbbb", lw=1.0, zorder=1)

box(0.6, 0.65, W - 1.2, 19.35, "#ffffff", "#2f2f2f", lw=1.8, r=0.3,
    ls=(0, (7, 4)), z=1)

# ---------------------------------------------------------------- 输入列
IN_X, IN_W = 0.95, 6.6
inputs = [
    ("Perception Stream", "monitoring video · frame stream"),
    ("Detection & Tracking", "keyframes · video clips"),
    ("Vessel Registry", "hull number · reference images"),
    ("Session History", "prior QA log · track index"),
]
in_y0, in_h, in_gap = 10.85, 1.42, 0.30
for i, (t1, t2) in enumerate(inputs):
    y = in_y0 + i * (in_h + in_gap)
    box(IN_X, y, IN_W, in_h, C["io"], C["io_e"], lw=1.1, r=0.16)
    txt(IN_X + 0.28, y + in_h - 0.45, t1, size=10.2, weight="bold", ha="left", family=CJK)
    txt(IN_X + 0.28, y + 0.42, t2, size=9.0, color=GREY, ha="left", family=CJK)
txt(IN_X, 17.62, "Inputs", size=9.4, weight="bold", color=GREY, ha="left", family=CJK)

# ---------------------------------------------------------------- User Query
Q_X, Q_Y, Q_W, Q_H = 12.6, 17.55, 19.4, 1.62
box(Q_X, Q_Y, Q_W, Q_H, C["accent"], C["accent_e"], lw=1.5, r=0.18)
txt(Q_X + 0.3, Q_Y + Q_H - 0.42, "User Query:", size=10.4, weight="bold",
    ha="left", family=CJK)
txt(Q_X + 0.3, Q_Y + 0.5,
    "\u201cDid Vessel 003 pass by the dock during this morning?\u201d",
    size=10.0, ha="left", family=CJK)

# ---------------------------------------------------------------- 三个子智能体
AG_Y, AG_H = 10.85, 4.75

P_X, P_W = 12.6, 8.2
box(P_X, AG_Y, P_W, AG_H, C["plan"], C["plan_e"], lw=1.6, r=0.24)
txt(P_X + P_W / 2, AG_Y + AG_H - 0.52, "Planning Agent", size=13, weight="bold", family=CJK)
ax.plot([P_X + 0.55, P_X + P_W - 0.55], [AG_Y + AG_H - 0.92] * 2,
        color=C["plan_e"], lw=0.9, zorder=3)
bullet_list(P_X + 0.45, AG_Y + AG_H - 1.44, [
    "· Intent Analysis Skill",
    "· Temporal Parsing Skill",
    "· Tool Planning Skill",
    "· Plan Generation Skill",
])
txt(P_X + P_W / 2, AG_Y + 0.5, "produces the initial plan (no tool calls)",
    size=9.2, color=GREY, family=CJK, style="italic")

E_X, E_W = 22.6, 10.4
box(E_X, AG_Y, E_W, AG_H, C["exec"], C["exec_e"], lw=2.2, r=0.24)
txt(E_X + E_W / 2, AG_Y + AG_H - 0.52, "Execution Agent", size=13.5,
    weight="bold", family=CJK)
txt(E_X + E_W / 2, AG_Y + AG_H - 0.94, "(primary worker)", size=8.8,
    color=GREY, style="italic", family=CJK)
ax.plot([E_X + 0.55, E_X + E_W - 0.55], [AG_Y + AG_H - 1.18] * 2,
        color=C["exec_e"], lw=0.9, zorder=3)
bullet_list(E_X + 0.45, AG_Y + AG_H - 1.64, [
    "· Tool Invocation Skill",
    "· Plan Execution Skill",
    "· Result Assembly Skill",
    "· State Synchronization Skill",
])
box(E_X + 0.45, AG_Y + 0.34, E_W - 0.9, 0.82, "#ffffff", C["exec_e"], lw=1.3, r=0.14, z=5)
txt(E_X + E_W / 2, AG_Y + 0.75, "TodoList middleware  →  write_todos",
    size=9.6, weight="bold", color=C["exec_e"], family=CJK)

R_X, R_W = 34.6, 8.2
box(R_X, AG_Y, R_W, AG_H, C["refl"], C["refl_e"], lw=1.6, r=0.24)
txt(R_X + R_W / 2, AG_Y + AG_H - 0.52, "Reflection Agent", size=13, weight="bold", family=CJK)
ax.plot([R_X + 0.55, R_X + R_W - 0.55], [AG_Y + AG_H - 0.92] * 2,
        color=C["refl_e"], lw=0.9, zorder=3)
bullet_list(R_X + 0.45, AG_Y + AG_H - 1.44, [
    "· Result Analysis Skill",
    "· Termination Determination",
    "· Evidence Aggregation Skill",
    "· Output Principle Skills",
])
txt(R_X + R_W / 2, AG_Y + 0.5, "exit only after acceptance check passes",
    size=9.2, color=GREY, family=CJK, style="italic")

# ---------------------------------------------------------------- 主流程箭头
MID = AG_Y + AG_H / 2
arrow((Q_X + Q_W / 2, Q_Y - 0.06), (P_X + P_W / 2, AG_Y + AG_H + 0.08), lw=2.2, ms=16)
arrow((IN_X + IN_W + 0.12, MID), (P_X - 0.12, MID), lw=2.6, ms=18)
label((IN_X + IN_W + P_X) / 2, MID, "state")

arrow((P_X + P_W + 0.1, MID), (E_X - 0.1, MID), lw=2.6, ms=18)
label((P_X + P_W + E_X) / 2, MID, "initial plan")

arrow((E_X + E_W + 0.12, MID - 0.55), (R_X - 0.12, MID - 0.55), lw=2.6, ms=18)
label((E_X + E_W + R_X) / 2, MID + 0.55, "result +")
label((E_X + E_W + R_X) / 2, MID + 0.05, "evidence IDs")

arrow((R_X + R_W / 2, AG_Y - 0.08), (E_X + E_W - 1.2, AG_Y - 0.08),
      color=C["refl_e"], lw=2.0, ms=16, rad=0.40, ls=(0, (5, 3)))
label((R_X + E_X + E_W) / 2 - 0.2, AG_Y - 1.55,
      "re-plan / re-execute (bounded rounds)", color=C["refl_e"])

# ---------------------------------------------------------------- 输出
O_X, O_Y, O_W, O_H = 44.4, AG_Y, 6.6, AG_H
box(O_X, O_Y, O_W, O_H, "#ffffff", "#2f2f2f", lw=1.6, r=0.24)
txt(O_X + O_W / 2, O_Y + O_H - 0.55, "Answer", size=13, weight="bold", family=CJK)
ax.plot([O_X + 0.4, O_X + O_W - 0.4], [O_Y + O_H - 0.95] * 2,
        color="#2f2f2f", lw=0.9, zorder=3)
txt(O_X + 0.32, O_Y + O_H - 1.55, "Vessel 003 first detected",
    size=9.4, ha="left", family=CJK)
txt(O_X + 0.32, O_Y + O_H - 2.02, "at 11:20–11:30 (confirmed).",
    size=9.4, ha="left", family=CJK)
box(O_X + 0.32, O_Y + 0.42, O_W - 0.64, 1.62, "#fff8e8", C["refl_e"],
    lw=1.1, r=0.14, z=5)
txt(O_X + O_W / 2, O_Y + 1.62, "evidence enriched by code", size=9.4,
    weight="bold", color=C["refl_e"], family=CJK)
txt(O_X + O_W / 2, O_Y + 1.05, "show_evidence · keyframes", size=9.0, color=GREY, family=CJK)
arrow((R_X + R_W + 0.1, MID), (O_X - 0.1, MID), lw=2.2, ms=16)

# ---------------------------------------------------------------- 记忆层
M_Y, M_H = 4.75, 2.55
memories = [
    ("Perception Memory", [
        "video memory (raw clips)",
        "annotated keyframe pool",
        "detector → tracker output",
    ]),
    ("Track Memory", [
        "track index (confirmed / uncertain)",
        "track-level evidence",
        "hull no. · confidence · keyframes",
    ]),
    ("System Memory", [
        "conversation memory",
        "vessel registry (reference images)",
        "feature embedding DB",
    ]),
]
mem_x0, mem_span, gap = 12.6, 38.4, 0.55
mw = (mem_span - 2 * gap) / 3
for i, (t, lines) in enumerate(memories):
    x = mem_x0 + i * (mw + gap)
    box(x, M_Y, mw, M_H, C["mem"], C["mem_e"], lw=1.2, r=0.18)
    txt(x + mw / 2, M_Y + M_H - 0.42, t, size=10.6, weight="bold", family=CJK)
    ax.plot([x + 0.4, x + mw - 0.4], [M_Y + M_H - 0.72] * 2,
            color=C["mem_e"], lw=0.8, zorder=3)
    for j, s in enumerate(lines):
        txt(x + 0.35, M_Y + M_H - 1.12 - j * 0.42, s, size=9.2,
            color="#333333", ha="left", family=CJK)
arrow((E_X + 1.4, M_Y + M_H + 0.08), (E_X + 1.4, AG_Y - 0.62),
      color=C["exec_e"], lw=1.8, ms=14, style="<|-|>")
label(E_X + 3.5, (M_Y + M_H + AG_Y) / 2 + 0.05, "read / write", color=C["exec_e"])

# ---------------------------------------------------------------- 工具层
T_Y, T_H = 1.75, 2.25
box(12.6, T_Y, 38.4, T_H, C["tool"], C["tool_e"], lw=1.2, r=0.18)
txt(12.6 + 0.35, T_Y + T_H - 0.42, "Domain Tools", size=10.6,
    weight="bold", ha="left", family=CJK)
tools = ["getTrack", "getFrames", "getClip", "getRegistry", "listRegistry",
         "matchHull", "matchText", "matchImage", "verifyTarget", "showEvidence",
         "dedupTracks", "read_file", "write_todos"]
# 7 列：内边距 1.55(左) + 0.45(右)，列宽 (38.4-2.0)/7
c0, cw = 12.6 + 1.05, (38.4 - 2.1) / 7
for i, t in enumerate(tools):
    col, row = i % 7, i // 7
    x = c0 + col * cw
    y = T_Y + T_H - 1.06 - row * 0.62
    box(x, y - 0.24, cw - 0.22, 0.48, "#ffffff", C["tool_e"], lw=0.9, r=0.1, z=5)
    txt(x + (cw - 0.22) / 2, y, t, size=9.2, family=CJK)
arrow((E_X + E_W / 2 - 1.4, T_Y + T_H + 0.06), (E_X + E_W / 2 - 1.4, AG_Y - 1.72),
      color=C["tool_e"], lw=1.6, ms=14, style="<|-|>")
arrow((R_X + R_W / 2 + 1.2, T_Y + T_H + 0.06), (R_X + R_W / 2 + 1.2, AG_Y - 1.72),
      color=C["tool_e"], lw=1.6, ms=14, style="<|-|>")

# ---------------------------------------------------------------- 边注
txt(0.95, 8.6, "Notes", size=9.6, weight="bold", color=GREY, ha="left", family=CJK)
txt(0.95, 7.9,
    "· plan is owned by the\n  Execution Agent (todos)\n"
    "· domain sub-agents now\n  live inside Execution\n"
    "· evidence is enriched and\n  rendered through code",
    size=9.0, color=GREY, ha="left", va="top", family=CJK)

# ---------------------------------------------------------------- 导出
base = r"G:\学业\论文\湖大论文4-agent\项目\SeaAgent\reference-figures\10-seaagent-3subagent-architecture"
fig.savefig(base + ".png", dpi=300, facecolor="white")
fig.savefig(base + ".svg", facecolor="white")
print("[ok]", base + ".png")
print("[ok]", base + ".svg")

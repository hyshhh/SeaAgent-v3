/* Sea-Video-Harness QA interaction layer. Public activity only; no hidden chain-of-thought is rendered. */

/* ── 会话与运行状态 ───────────────────────────────────────────────────────────
   一个会话可保存多条 QA 记录，每条 QA 独立运行。sessionRuns 以 session_id 为键保存当前运行：
   事件先写进该轮的缓冲区，只有当前正在查看这个会话时才落到 DOM 上。于是：

     · 切到别的会话，运行继续跑，事件一条不丢；
     · 切回来立刻看到这一轮的完整实时过程（缓冲区回放），而不是空白；
     · 正在跑的会话在列表里带「回答中」标记，随时可以切走或停止。

   currentSessionId 只表示「正在看哪个会话」，不再兼任运行归属。 */
const sessionRuns = new Map();
let currentSessionId = null;   // 正在查看的会话；null = 尚未提问的新会话
let currentSession = null;     // 该会话在服务端的存档（含每轮问答）
let sessionSummaries = [];     // 会话栏列表

let harnessResult = null;
let harnessEvidence = null;
let harnessToolCount = 0;
let harnessIteration = 0;
let harnessStageSummary = '等待问答';
let harnessLastStateChange = '—';
const harnessAgentStates = new Map();
const harnessToolCards = new Map();
const harnessSkillNames = new Set();
let harnessSkillActivityCard = null;
let qaView = 'chat';           // 主列当前显示的子页：对话、轨迹或实时状态

const AGENT_LABELS = { planner: 'Planner', executor: 'Executor', reflector: 'Reflector' };
const AGENT_STATE_LABELS = {
  pending: '待执行', running: '执行中', waiting: '等待中', completed: '已完成',
  failed: '失败', cancelled: '已停止', reconnecting: '重连中', unknown: '未记录',
};
const NON_BUSINESS_TOOLS = new Set([
  'read_shared_state', 'update_planner_state', 'update_execution_state',
  'update_reflection_state', 'read_file',
]);

const STEP_KINDS = {
  tool: { label: '工具', dot: 'qa-dot-tool' },
  skill: { label: '技能', dot: 'qa-dot-skill' },
  system: { label: '状态', dot: 'qa-dot-system' },
  error: { label: '错误', dot: 'qa-dot-error' },
};

function useQuestion(text) {
  const input = document.getElementById('agentQuestion');
  if (input) {
    input.value = text;
    input.focus();
    autoGrowComposer();
  }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function compact(value, limit = 600) {
  let text;
  try { text = typeof value === 'string' ? value : JSON.stringify(value ?? '', null, 2); }
  catch (_error) { text = String(value ?? ''); }
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

function formatEventTime() {
  return new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
}

function renderAgentStatus() {
  const iteration = document.getElementById('qaIterationCount');
  const stage = document.getElementById('qaStageSummary');
  const changed = document.getElementById('qaLastStateChange');
  if (iteration) iteration.textContent = String(harnessIteration);
  if (stage) stage.textContent = harnessStageSummary;
  if (changed) changed.textContent = harnessLastStateChange;
  for (const [agent, value] of harnessAgentStates) {
    const card = document.querySelector(`.qa-agent-status[data-agent="${agent}"]`);
    if (!card) continue;
    card.dataset.state = value.state;
    card.querySelector('.qa-agent-status-label').textContent = AGENT_STATE_LABELS[value.state] || value.state;
    card.querySelector('p').textContent = value.action;
  }
}

function setAgentStatus(agent, state, action, iteration) {
  const role = String(agent || '');
  const nextState = String(state || 'running');
  const message = String(action || '正在处理');
  if (Number.isFinite(Number(iteration))) harnessIteration = Math.max(harnessIteration, Number(iteration));
  if (harnessAgentStates.has(role)) harnessAgentStates.set(role, {state: nextState, action: message});
  harnessStageSummary = `${AGENT_LABELS[role] || role || 'Harness'} · ${message}`;
  harnessLastStateChange = formatEventTime();
  renderAgentStatus();
}

/* 会话时间戳由后端给 ISO-8601（带时区）；旧格式会话没有时间戳，返回空串让调用方省掉这一段。 */
function formatSessionTime(value) {
  const date = new Date(String(value || ''));
  if (!value || Number.isNaN(date.getTime())) return '';
  const pad = (number) => String(number).padStart(2, '0');
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/* 会话条目的副标题：有时间的显示「时间 · N 轮」，没有轮次的（如正在回答的新会话）只留有时间的部分。 */
function sessionMeta(session) {
  const stamp = formatSessionTime(session?.updatedAt);
  const turns = Number(session?.turnCount || 0);
  return [stamp, turns > 0 ? `${turns} 轮` : ''].filter(Boolean).join(' · ');
}

/* 会话号由前端生成，运行一开始就有 key，不必等后端回传才开始记账。 */
function newSessionId() {
  const bytes = new Uint8Array(6);
  if (window.crypto?.getRandomValues) window.crypto.getRandomValues(bytes);
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `session-${hex || Math.random().toString(16).slice(2, 14)}`;
}

function setHarnessState(label, state = '') {
  const labelNode = document.getElementById('agentThinkingState');
  const intentNode = document.getElementById('agentIntentState');
  const inspectorNode = document.getElementById('qaInspectorState');
  const headerNode = document.getElementById('qaHeaderStatus');
  const panel = document.getElementById('agentResultPanel');
  const headerDot = document.getElementById('qaHeaderDot');
  const thinkingDot = document.getElementById('agentThinkingDot');
  if (labelNode) labelNode.textContent = label;
  if (intentNode) intentNode.textContent = label;
  if (inspectorNode) inspectorNode.textContent = label;
  if (headerNode) headerNode.textContent = label;
  if (panel) panel.dataset.state = state === 'complete' ? 'completed' : state === 'failed' ? 'failed' : state || 'idle';
  [headerDot, thinkingDot].forEach((node) => { if (node) node.className = state; });
}

function updateHarnessStats() {
  const toolNode = document.getElementById('qaToolCount');
  const skillNode = document.getElementById('qaSkillCount');
  if (toolNode) toolNode.textContent = String(harnessToolCount);
  if (skillNode) skillNode.textContent = String(harnessSkillNames.size);
}

function isNearBottom(node) {
  return !node || node.scrollHeight - node.scrollTop - node.clientHeight < 120;
}

function scrollConversation(force = false) {
  const node = document.getElementById('qaConversationScroll');
  if (!node || (!force && !isNearBottom(node))) return;
  requestAnimationFrame(() => { node.scrollTop = node.scrollHeight; });
}

function scrollActivity(force = false) {
  const node = document.getElementById('agentThoughtStream');
  if (!node || (!force && !isNearBottom(node))) return;
  requestAnimationFrame(() => { node.scrollTop = node.scrollHeight; });
}

/* 清空「当前这一轮」的渲染状态；切会话、重放事件前都会调用，保证渲染是幂等的。 */
function resetActivityDom() {
  renderAgentPlan({ steps: [] });
  closeAgentConfirmation('idle');
  const stream = document.getElementById('agentActivityStream');
  if (stream) stream.innerHTML = '<div class="qa-empty-state">Activity will appear here when the harness starts.</div>';
  const final = document.getElementById('agentFinalView');
  if (final) final.hidden = true;
  const skills = document.getElementById('qaSkillList');
  if (skills) skills.innerHTML = '<div class="qa-inspector-empty">Skills load with the harness.</div>';
  const evidence = document.getElementById('evidenceGallery');
  if (evidence) evidence.innerHTML = '<div class="qa-inspector-empty">No evidence available.</div>';
  const count = document.getElementById('evidenceResultCount');
  if (count) count.textContent = 'Waiting';
  harnessIteration = 0;
  harnessStageSummary = '等待问答';
  harnessLastStateChange = '—';
  harnessAgentStates.clear();
  for (const agent of Object.keys(AGENT_LABELS)) {
    harnessAgentStates.set(agent, { state: 'pending', action: '等待任务' });
  }
  harnessToolCount = 0;
  agentLines.clear();  // 新一轮开始：子智能体行重新来
  harnessToolCards.clear();
  harnessSkillNames.clear();
  harnessSkillActivityCard = null;
  harnessResult = null;
  harnessEvidence = null;
  updateHarnessStats();
  renderAgentStatus();
}

/* 输入框默认一行，随内容长高到上限为止（不用 CSS 的 field-sizing，兼容性还不齐）。 */
function autoGrowComposer() {
  const node = document.getElementById('agentQuestion');
  if (!node) return;
  node.style.height = 'auto';
  node.style.height = `${Math.min(node.scrollHeight, 132)}px`;
}

/* 输入区的按钮随「当前会话是否在跑」切换：跑着的时候只能停，不能在同一个会话里再发一轮。 */
function syncComposer(run) {
  const button = document.getElementById('btnAskAgent');
  const stop = document.getElementById('btnStopAgent');
  const running = run?.status === 'running';
  const awaiting = run?.status === 'awaiting_confirmation';
  if (button) {
    button.disabled = running || awaiting;
    button.title = running ? '本轮进行中…' : awaiting ? '请先批准或拒绝当前操作' : '运行（Ctrl / ⌘ + Enter）';
  }
  if (stop) {
    stop.hidden = !running;
    stop.disabled = !running;
  }
}

/* ── 子页：对话 / 轨迹 ─────────────────────────────────────────────────────── */

function switchQaView(view) {
  qaView = ['trace', 'state'].includes(view) ? view : 'chat';
  const chat = document.getElementById('qaChatView');
  const trace = document.getElementById('qaTraceView');
  const state = document.getElementById('qaStateView');
  const chatTab = document.getElementById('qaSubtabChat');
  const traceTab = document.getElementById('qaSubtabTrace');
  const stateTab = document.getElementById('qaSubtabState');
  if (chat) chat.hidden = qaView !== 'chat';
  if (trace) trace.hidden = qaView !== 'trace';
  if (state) state.hidden = qaView !== 'state';
  chatTab?.classList.toggle('is-active', qaView === 'chat');
  traceTab?.classList.toggle('is-active', qaView === 'trace');
  stateTab?.classList.toggle('is-active', qaView === 'state');
  chatTab?.setAttribute('aria-selected', String(qaView === 'chat'));
  traceTab?.setAttribute('aria-selected', String(qaView === 'trace'));
  stateTab?.setAttribute('aria-selected', String(qaView === 'state'));
  if (pendingConfirmation) mountConfirmationPanel();
  if (qaView === 'trace') renderTrajectory();
  else if (qaView === 'state') renderAgentStatus();
  else { scrollConversation(true); autoGrowComposer(); }
}

/* 轨迹的数据来源：优先当前这一轮的事件缓冲（有参数与结果），退回会话存档（只有工具名）。 */
function trajectorySteps() {
  const run = currentSessionId ? sessionRuns.get(currentSessionId) : null;
  if (run) {
    // 打开历史轮次时把最后的计划快照重画出来：事件缓冲里有本轮全部 plan 事件
    const plans = run.events.filter((event) => event.type === 'plan');
    if (plans.length) renderAgentPlan(plans[plans.length - 1]);
    return { steps: stepsFromEvents(run.events), archived: false };
  }
  const turns = Array.isArray(currentSession?.turns) ? currentSession.turns : [];
  const latest = turns[turns.length - 1];
  renderAgentPlan(latest?.plan || { steps: [] });
  return latest ? { steps: stepsFromTurn(latest), archived: true } : { steps: [], archived: false };
}

/* ---------------------------------------------------------------------------
 * 共享计划与执行进度由运行时以 plan 事件广播，按完整快照重画。
 * ------------------------------------------------------------------------- */
const PLAN_STATUS_LABEL = {
  pending: '待办',
  in_progress: '进行中',
  completed: '已完成',
  failed: '失败',
};

function renderAgentPlan(event) {
  const panel = document.getElementById('agentPlanPanel');
  const list = document.getElementById('agentPlanList');
  const progress = document.getElementById('agentPlanProgress');
  const fill = document.getElementById('agentPlanMeterFill');
  if (!panel || !list) return;
  const steps = Array.isArray(event.steps) ? event.steps : [];
  panel.hidden = steps.length === 0;
  list.innerHTML = steps.map((item) => {
    const status = String(item.status || 'pending');
    const label = PLAN_STATUS_LABEL[status] || status;
    const mark = status === 'completed' ? '✓' : status === 'in_progress' ? '▸' : '·';
    return `<li class="qa-plan-item" data-status="${escapeHtml(status)}"><span class="qa-plan-mark" aria-hidden="true">${mark}</span><span class="qa-plan-text">${escapeHtml(item.content || '')}</span><em class="qa-plan-status">${escapeHtml(label)}</em></li>`;
  }).join('');
  const total = Number(event.total ?? steps.length);
  const done = Number(event.completed ?? steps.filter((item) => item.status === 'completed').length);
  if (progress) progress.textContent = `${done}/${total} 步已完成`;
  if (fill) fill.style.width = total ? `${Math.round((done / total) * 100)}%` : '0%';
  panel.dataset.state = done === total ? 'done' : 'running';
}

/* ---------------------------------------------------------------------------
 * 写入确认卡：executor 调 add_registry_vessel 时被 HumanInTheLoop 中断拦下，
 * 运行时以 confirm 事件把载荷推过来。这里只做展示与回传——
 * 批准/拒绝之后由后端在同一个 thread 上恢复，本轮继续往下跑。
 * ------------------------------------------------------------------------- */
let pendingConfirmation = null;

function mountConfirmationPanel() {
  const panel = document.getElementById('agentConfirmPanel');
  const mount = document.getElementById(
    qaView === 'trace' ? 'qaTraceConfirmationMount' : 'qaChatConfirmationMount',
  );
  if (panel && mount && panel.parentElement !== mount) mount.appendChild(panel);
  if (panel && !panel.hidden && qaView === 'chat') scrollConversation(true);
}

const CONFIRM_FIELD_LABEL = {
  hull_number: '舷号',
  description: '描述',
  image_paths: '参考图',
  aliases: '别名',
  user_intent: '用户要求',
};

function renderAgentConfirmation(event) {
  const panel = document.getElementById('agentConfirmPanel');
  const body = document.getElementById('agentConfirmBody');
  const hint = document.getElementById('agentConfirmHint');
  if (!panel || !body) return;
  const confirmation = event.confirmation || {};
  const actions = Array.isArray(confirmation.actions) ? confirmation.actions : [];
  const action = actions[0] || {};
  if (!action.tool) return;
  mountConfirmationPanel();
  pendingConfirmation = { interruptId: confirmation.interruptId || '', sessionId: currentSessionId || '' };
  const rows = Object.entries(action.arguments || {}).map(([key, value]) => {
    const label = CONFIRM_FIELD_LABEL[key] || key;
    const text = Array.isArray(value) ? value.join('、') : String(value ?? '');
    return `<div class="qa-confirm-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(text)}</dd></div>`;
  }).join('');
  body.innerHTML = rows;
  if (hint) hint.textContent = action.description || `工具：${action.tool}`;
  const feedback = document.getElementById('agentConfirmFeedback');
  if (feedback) { feedback.value = ''; feedback.disabled = false; }
  for (const id of ['btnConfirmApprove', 'btnConfirmReject']) {
    const button = document.getElementById(id);
    if (button) button.disabled = false;
  }
  panel.hidden = false;
  panel.dataset.state = 'pending';
  if (qaView === 'chat') scrollConversation(true);
  else panel.scrollIntoView?.({ block: 'nearest' });
}

function closeAgentConfirmation(state) {
  const panel = document.getElementById('agentConfirmPanel');
  if (!panel) return;
  panel.dataset.state = state;
  panel.hidden = true;
  pendingConfirmation = null;
}

async function submitConfirmation(decision) {
  if (!pendingConfirmation || !pendingConfirmation.sessionId) return;
  const feedback = document.getElementById('agentConfirmFeedback');
  const panel = document.getElementById('agentConfirmPanel');
  if (panel?.dataset.state === 'sending') return;
  const sessionId = pendingConfirmation.sessionId;
  const run = sessionRuns.get(sessionId) || {
    id: sessionId, question: currentSession?.turns?.at(-1)?.question || '', events: [], result: null,
  };
  if (run.status === 'running') return;
  run.status = 'running';
  sessionRuns.set(sessionId, run);
  syncComposer(run);
  if (panel) panel.dataset.state = 'sending';
  for (const id of ['btnConfirmApprove', 'btnConfirmReject']) {
    const button = document.getElementById(id);
    if (button) button.disabled = true;
  }
  try {
    const response = await fetch('/api/agent/query/resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId,
        decision,
        feedback: decision === 'reject' ? String((feedback && feedback.value) || '').trim() : '',
      }),
    });
    if (!response.ok) throw new Error(`恢复失败：${response.status}`);
    const result = await response.json();
    run.result = result;
    run.status = result.state === 'awaiting_confirmation' ? 'awaiting_confirmation' : result.state === 'cancelled' ? 'stopped' : result.success === false ? 'error' : 'done';
    run.events = run.events.filter((event) => event.type !== 'confirm');
    if (currentSessionId === sessionId) {
      closeAgentConfirmation(decision === 'approve' ? 'approved' : 'rejected');
      renderAgentAnswer(result);
      syncComposer(run);
    }
    await loadSessions();
  } catch (error) {
    run.status = 'awaiting_confirmation';
    if (currentSessionId === sessionId) {
      if (panel) panel.dataset.state = 'failed';
      for (const id of ['btnConfirmApprove', 'btnConfirmReject']) {
        const button = document.getElementById(id);
        if (button) button.disabled = false;
      }
      syncComposer(run);
      appendStandardEvent({ title: '恢复失败' }, 'error', '!', 'CONFIRM', String(error && error.message ? error.message : error));
    }
  }
}

function bindConfirmationButtons() {
  const approve = document.getElementById('btnConfirmApprove');
  const reject = document.getElementById('btnConfirmReject');
  if (approve) approve.addEventListener('click', () => submitConfirmation('approve'));
  if (reject) reject.addEventListener('click', () => submitConfirmation('reject'));
}

function stepsFromEvents(events) {
  const steps = [];
  const byCallId = new Map();
  const push = (kind, text, extra = {}) => { steps.push({ kind, text, ...extra }); return steps[steps.length - 1]; };
  const rounds = new Set();
  for (const event of events) {
    const agent = String(event.agent || '');
    const iteration = Number(event.iteration || event.round || harnessIteration || 1);
    if (event.type === 'plan') renderAgentPlan(event);
    if (Number.isFinite(iteration) && !rounds.has(iteration) && event.type === 'agent_status' && event.state === 'running') {
      rounds.add(iteration);
      push('system', `第 ${iteration} 轮开始`);
    }
    if (event.type === 'tool_start' || event.type === 'tool_result') {
      const tool = String(event.tool || '');
      if (!tool || tool === 'task') continue;
      const stateChange = tool === 'read_shared_state' || tool.startsWith('update_');
      const skillRead = tool === 'read_file';
      const kind = stateChange ? 'system' : skillRead ? 'skill' : 'tool';
      const label = stateChange ? stateToolLabel(tool) : skillRead ? 'Read skill' : event.label || tool;
      const callId = String(event.callId || '');
      if (event.type === 'tool_start') {
        const step = push(kind, label, {
          tool,
          args: event.arguments,
          status: 'running',
          agent,
          iteration,
        });
        byCallId.set(callId, step);
      } else {
        const step = byCallId.get(callId);
        const failed = event.status === 'error' || event.result?.ok === false;
        if (step) step.status = failed ? 'error' : 'ok';
        else push(kind, label, {tool, status: failed ? 'error' : 'ok', agent, iteration});
      }
    } else if (event.type === 'confirm') {
      renderAgentConfirmation(event);
      const action = event.confirmation?.actions?.[0];
      const pending = [...steps].reverse().find((step) =>
        step.kind === 'tool' && step.status === 'running' && step.tool === action?.tool,
      );
      if (pending) pending.status = 'awaiting_confirmation';
    } else if (event.type === 'error') {
      push('error', event.message || '执行失败', {agent, iteration});
    }
  }
  return steps;
}

function stateToolLabel(tool) {
  if (tool === 'read_shared_state') return 'Read shared state';
  return 'Update shared state';
}

/* 会话存档只留了工具名序列：参数与结果属于当轮事件流，这里如实说明，不假装有细节。 */
function stepsFromTurn(turn) {
  return turn ? [{kind: 'system', text: '历史记录仅保存问答内容，不保留运行事件。'}] : [];
}

function renderTrajectory() {
  const list = document.getElementById('qaTraceList');
  const ribbon = document.getElementById('qaTraceRibbon');
  const summary = document.getElementById('qaTraceSummary');
  if (!list) return;
  const { steps, archived } = trajectorySteps();
  const toolCount = steps.filter((step) => step.kind === 'tool').length;
  const failed = steps.filter((step) => step.status === 'error').length;
  const round = harnessIteration || (archived ? currentSession?.turns?.length || 0 : 0);
  if (summary) {
    summary.textContent = `第 ${round} 轮 · ${toolCount} 次业务工具${failed ? ` · ${failed} 次失败` : ''}${archived ? ' · 历史记录' : ''}`;
  }
  if (ribbon) {
    ribbon.innerHTML = steps.map((step) => `<i class="${STEP_KINDS[step.kind]?.dot || 'qa-dot-system'}" title="${escapeHtml(STEP_KINDS[step.kind]?.label || '')}"></i>`).join('');
  }
  if (!steps.length) {
    list.innerHTML = '<div class="qa-empty-state">这一轮开始时，助手与工具的每一步都会记在这里。</div>';
    return;
  }
  list.innerHTML = steps.map((step, index) => {
    const kind = STEP_KINDS[step.kind] || STEP_KINDS.system;
    const badge = step.status === 'awaiting_confirmation' ? '<span class="qa-step-badge is-confirmation">等待人工确认</span>'
      : step.status === 'running' ? '<span class="qa-step-badge is-running">执行中</span>'
      : step.status === 'error' ? '<span class="qa-step-badge is-error">失败</span>'
      : step.status === 'ok' ? '<span class="qa-step-badge is-ok">完成</span>' : '';
    // 从智能体做的步骤降一级并挂上名字，主从结构在轨迹里一眼可见
    const owner = step.agent ? `<span class="qa-step-agent">${escapeHtml(step.agent)}</span>` : '';
    const delegation = step.delegation ? `<span class="qa-step-delegation">→ ${escapeHtml(step.delegation)}</span>` : '';
    const args = step.args === undefined ? '' : escapeHtml(summarizeToolArguments(step.tool, step.args));
    const iteration = step.iteration ? `<span class="qa-step-round">第 ${step.iteration} 轮</span>` : '';
    return `<article class="qa-step${step.agent ? ' is-subagent' : ''}">
      <span class="qa-step-kind"><i class="${kind.dot}"></i>${kind.label}</span>
      <div class="qa-step-body">
        <div class="qa-step-line">${iteration}${owner}<strong>${escapeHtml(step.text || '')}</strong>${delegation}${badge}</div>
        ${args ? `<div class="qa-step-args">${args}</div>` : ''}
      </div>
    </article>`;
  }).join('');
}

function renderViewMeta() {
  const node = document.getElementById('qaViewMeta');
  if (!node) return;
  const session = currentSession;
  const turns = Array.isArray(session?.turns) ? session.turns.length : 0;
  const run = currentSessionId ? sessionRuns.get(currentSessionId) : null;
  const title = session?.title || run?.question || '';
  const round = turns + (run && run.status === 'running' ? 1 : 0);
  node.textContent = title ? `${title}${round ? ` · 第 ${round} 轮` : ''}` : '新会话';
}

/* ── 会话栏 ───────────────────────────────────────────────────────────────── */

function renderSessionList() {
  const list = document.getElementById('qaSessionList');
  const count = document.getElementById('qaSessionCount');
  if (count) count.textContent = String(sessionSummaries.length);
  if (!list) return;
  if (!sessionSummaries.length) {
    list.innerHTML = '<div class="qa-sessions-empty">还没有会话。提一个问题，对话就会出现在这里。</div>';
    return;
  }
  list.innerHTML = sessionSummaries.map((session) => {
    const active = session.sessionId === currentSessionId ? ' is-active' : '';
    const title = session.title || '未命名会话';
    const live = session.running ? '<span class="qa-session-live">● 回答中</span>' : '';
    return `<div class="qa-session-item${active}" role="button" tabindex="0" data-session-id="${escapeHtml(session.sessionId)}" onclick="openSession(this.dataset.sessionId)" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openSession(this.dataset.sessionId);}">
      <span class="qa-session-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9.5 9.5 0 0 1-3.3-.6L3 21l1.7-4.6A8.3 8.3 0 0 1 3.6 11.5 8.4 8.4 0 0 1 12 3.1a8.4 8.4 0 0 1 9 8.4Z"/></svg></span>
      <div class="qa-session-body">
        <strong class="qa-session-name">${escapeHtml(title)}</strong>
        <small class="qa-session-note">${live}${live ? ' · ' : ''}${escapeHtml(sessionMeta(session))}</small>
      </div>
      <button type="button" class="qa-session-delete" title="删除该会话" aria-label="删除该会话" onclick="event.stopPropagation();deleteSession(this.closest('.qa-session-item').dataset.sessionId)">×</button>
    </div>`;
  }).join('');
}

async function loadSessions() {
  let server = [];
  try {
    const response = await fetch('/api/agent/sessions');
    const data = await response.json();
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    server = Array.isArray(data?.sessions) ? data.sessions : [];
  } catch (_error) {
    server = [];
  }
  // 没有标题也没有轮次的会话点开就是空白，不进列表（服务端启动时也会清掉这类残壳）
  server = server.filter((item) => item.title || Number(item.turnCount || 0) > 0);
  // 合并本地正在跑的会话：服务端可能刚建行、也可能还没轮到写列表
  const byId = new Map(server.map((item) => [item.sessionId, item]));
  for (const [id, run] of sessionRuns) {
    const entry = byId.get(id) || { sessionId: id, title: run.question, updatedAt: '', turnCount: 0 };
    entry.running = run.status === 'running';
    byId.set(id, entry);
  }
  sessionSummaries = [...byId.values()].sort((left, right) => {
    if (!!right.running !== !!left.running) return right.running ? 1 : -1;
    return String(right.updatedAt || '').localeCompare(String(left.updatedAt || ''));
  });
  renderSessionList();
  return sessionSummaries;
}

async function fetchSession(sessionId) {
  if (!sessionId) return null;
  const response = await fetch(`/api/agent/sessions/${encodeURIComponent(sessionId)}`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
  return data;
}

/* 已往轮次只展示问答对：工具明细属于当轮实时事件流，历史里保留工具名序列足够定位。 */
function renderSessionHistory(turns) {
  const node = document.getElementById('qaSessionHistory');
  if (!node) return;
  const past = Array.isArray(turns) ? turns : [];
  if (!past.length) {
    node.hidden = true;
    node.innerHTML = '';
    return;
  }
  const shown = past.slice(-6);
  const folded = past.length - shown.length;
  const head = folded > 0 ? `<div class="qa-history-more">已折叠较早的 ${folded} 轮对话</div>` : '';
  node.innerHTML = head + shown.map((turn) => `<article class="qa-history-item">
      <span>You</span>
      <p>${escapeHtml(turn.question || '')}</p>
      <span>Harness</span>
      <p class="qa-history-answer">${escapeHtml(compact(turnAnswer(turn), 420))}</p>
    </article>`).join('');
  node.hidden = false;
}

function turnAnswer(turn) {
  if (turn?.answer) return turn.answer;
  if (turn?.state === 'cancelled') return '（这一轮已停止）';
  if (turn?.state === 'stalled') return '（这一轮未完成，未生成有效结论）';
  return turn?.state || '（该轮未留下回答）';
}

/* 把某一轮问答还原到主面板；会话记录只恢复问题和回答。 */
function renderRestoredTurn(turn) {
  renderAgentPlan({ steps: [] });
  harnessIteration = currentSession?.turns?.length || 0;
  harnessStageSummary = '历史记录 · 运行状态未保存';
  for (const agent of Object.keys(AGENT_LABELS)) {
    harnessAgentStates.set(agent, {state: 'unknown', action: '历史记录不保留运行状态'});
  }
  if (turn?.state === 'awaiting_confirmation') {
    harnessAgentStates.set('executor', {state: 'waiting', action: '等待人工确认'});
    harnessStageSummary = '等待人工确认';
    renderAgentStatus();
    renderAgentConfirmation({ confirmation: turn.confirmation });
    syncComposer({ status: 'awaiting_confirmation' });
    setHarnessState('等待确认', 'awaiting_confirmation');
    return;
  }
  const view = document.getElementById('agentFinalView');
  const answer = document.getElementById('agentAnswer');
  if (view) view.hidden = false;
  if (answer) answer.textContent = turnAnswer(turn) || '（该轮未留下回答）';
  const failed = turn?.state === 'error' || turn?.state === 'stalled';
  const awaitingInput = turn?.state === 'awaiting_input';
  const state = awaitingInput ? '需要补充信息' : turn?.state === 'stalled' ? '未完成' : failed ? 'Failed' : turn?.state === 'cancelled' ? 'Stopped' : turn?.partial ? '已按现有证据收尾' : 'Restored';
  renderAgentStatus();
  setHarnessState(state, failed ? 'failed' : turn?.state === 'cancelled' ? '' : 'complete');
}

/* 一轮结束后由运行结果收尾；客户端提前断开时用占位结果补齐。 */
function renderRunOutcome(run) {
  if (run.result) {
    renderAgentAnswer(run.result);
    return;
  }
  const stopped = run.status === 'stopped';
  renderAgentAnswer({
    success: false,
    state: stopped ? 'cancelled' : 'error',
    answerText: stopped ? '本轮已停止。' : (run.error || '本轮未能完成。'),
    evidence: {},
    toolRecords: [],
    toolChain: [],
  });
}

/* 主面板的唯一渲染入口：正在看的会话 + 它当前那一轮（运行中 / 已结束 / 存档）。 */
function renderView() {
  const session = currentSession;
  const run = currentSessionId ? sessionRuns.get(currentSessionId) : null;
  const turns = Array.isArray(session?.turns) ? session.turns : [];
  // 运行结果里带着轮次序号：等于已存轮数说明这一轮已经落库，历史里就要去掉它，免得同一轮显示两次
  const stored = !!(run?.result && Number(run.result.turnIndex) === turns.length && turns.length > 0);
  const history = run ? (stored ? turns.slice(0, -1) : turns) : turns.slice(0, -1);
  const latest = turns[turns.length - 1] || null;

  resetActivityDom();
  syncComposer(run);
  renderSessionHistory(history);
  renderViewMeta();
  if (qaView === 'trace') renderTrajectory();

  const welcome = document.getElementById('qaWelcome');
  const userTurn = document.getElementById('qaUserTurn');
  if (!session && !run) {
    if (welcome) welcome.hidden = false;
    if (userTurn) userTurn.hidden = true;
    setHarnessState('Ready', '');
    return;
  }
  if (welcome) welcome.hidden = true;
  if (userTurn) userTurn.hidden = false;
  const questionNode = document.getElementById('qaUserQuestion');
  if (questionNode) questionNode.textContent = run ? run.question : (latest?.question || '');

  if (run) {
    run.events.forEach((event) => appendHarnessEvent(event));
    if (run.status === 'running') setHarnessState('Running', 'running');
    else renderRunOutcome(run);
  } else if (latest) {
    renderRestoredTurn(latest);
  }
  scrollActivity(true);
  scrollConversation(true);
}

async function openSession(sessionId) {
  if (!sessionId || sessionId === currentSessionId) return;
  currentSessionId = sessionId;
  const run = sessionRuns.get(sessionId);
  // 有运行就先按运行渲染——即便它这一轮还没落库，也绝不会是空白
  if (!currentSession || currentSession.sessionId !== sessionId) {
    currentSession = { sessionId, title: run?.question || '', turns: [] };
  }
  renderSessionList();
  renderView();
  try {
    const detail = await fetchSession(sessionId);
    if (currentSessionId !== sessionId) return;   // 期间又切走了，别覆盖新视图
    currentSession = detail;
    renderView();
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

function startNewSession() {
  // 只把视图切到空白会话，不动正在后台跑的那一轮
  currentSessionId = null;
  currentSession = null;
  renderSessionList();
  renderView();
  const welcome = document.getElementById('qaWelcome');
  if (welcome) welcome.hidden = false;
  const userTurn = document.getElementById('qaUserTurn');
  if (userTurn) userTurn.hidden = true;
  document.getElementById('agentQuestion')?.focus();
  scrollConversation(true);
}

async function deleteSession(sessionId) {
  if (!sessionId) return;
  try {
    const response = await fetch(`/api/agent/sessions/${encodeURIComponent(sessionId)}`, {method: 'DELETE'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    sessionRuns.get(sessionId)?.controller?.abort();
    sessionRuns.delete(sessionId);
    if (currentSessionId === sessionId) {
      currentSessionId = null;
      currentSession = null;
      renderView();
    }
    await loadSessions();
    if (typeof showToast === 'function') showToast('会话已删除');
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

/* ── 事件渲染 ─────────────────────────────────────────────────────────────── */

function appendSkillToInspector(event) {
  const name = String(event.skill || event.name || event.title || '').trim();
  if (!name || harnessSkillNames.has(name)) return;
  harnessSkillNames.add(name);
  const list = document.getElementById('qaSkillList');
  if (!list) return;
  list.querySelector('.qa-inspector-empty')?.remove();
  const item = document.createElement('div');
  item.className = 'qa-skill-item';
  item.innerHTML = `<i aria-hidden="true">✦</i><div><strong>${escapeHtml(name)}</strong><span>${escapeHtml(compact(event.message || 'Skill loaded', 92))}</span></div>`;
  list.appendChild(item);
}

function appendStandardEvent(event, kind, icon, label, message) {
  const stream = document.getElementById('agentActivityStream');
  if (!stream) return;
  stream.querySelector('.qa-empty-state')?.remove();
  const row = document.createElement('article');
  row.className = `qa-event-row qa-${kind}`;
  row.innerHTML = `<span class="qa-event-icon" aria-hidden="true">${icon}</span><div class="qa-event-main"><div class="qa-event-title"><span>${escapeHtml(event.title || label)}</span><em class="qa-event-label">${escapeHtml(label)}</em></div><div class="qa-event-message">${escapeHtml(message || '')}</div></div><time class="qa-event-meta">${formatEventTime()}</time>`;
  stream.appendChild(row);
  return row;
}

function updateSkillActivitySummary() {
  const names = [...harnessSkillNames];
  const message = names.length
    ? `已挂载 ${names.length} 个 Skills · ${names.join('、')}`
    : 'Skills 已挂载';
  if (!harnessSkillActivityCard) {
    harnessSkillActivityCard = appendStandardEvent({title: 'Skills 已注入'}, 'skill', '✦', 'SKILLS', message);
    return;
  }
  const messageNode = harnessSkillActivityCard.querySelector('.qa-event-message');
  if (messageNode) messageNode.textContent = message;
}

function createToolEvent(event) {
  const stream = document.getElementById('agentActivityStream');
  if (!stream) return null;
  stream.querySelector('.qa-empty-state')?.remove();
  const callId = String(event.callId || `${Date.now()}-${harnessToolCount}`);
  if (harnessToolCards.has(callId)) return harnessToolCards.get(callId);
  const tool = String(event.label || event.tool || 'Tool');
  const item = document.createElement('article');
  item.className = 'qa-tool-event is-running';
  item.dataset.callId = callId;
  const args = summarizeToolArguments(event.tool, event.arguments || {});
  const owner = AGENT_LABELS[event.agent] || '';
  item.innerHTML = `<div class="qa-tool-event-head"><span class="qa-tool-dot" aria-hidden="true"></span><span class="qa-tool-name">${escapeHtml(tool)}</span>${owner ? `<span class="qa-tool-agent">${escapeHtml(owner)}</span>` : ''}<span class="qa-tool-args">${escapeHtml(args)}</span><span class="qa-tool-status">Running</span></div>`;
  stream.appendChild(item);
  harnessToolCards.set(callId, item);
  return item;
}

function summarizeToolArguments(name, value) {
  const args = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const summaryValue = (item) => {
    if (Array.isArray(item)) return `${item.length} items`;
    if (item && typeof item === 'object') return `${Object.keys(item).length} fields`;
    const text = String(item ?? '');
    return text.length > 72 ? `${text.slice(0, 72)}…` : text;
  };
  if (name === 'read_shared_state') return 'shared state';
  if (name === 'read_file') return String(args.file_path || args.path || 'skill file');
  if (name?.startsWith('update_')) {
    const patch = args.patch && typeof args.patch === 'object' ? args.patch : args;
    const fields = Object.entries(patch).map(([key, item]) => {
      if (key === 'plan' || key === 'requirements') return `${key}: ${item.length || 0}`;
      if (key === 'step_status') return `steps: ${Object.entries(item || {}).map(([id, state]) => `${id} ${state}`).join(', ')}`;
      if (key === 'assessment') return 'assessment updated';
      if (key === 'evidence_gaps' || key === 'conflicts') return `${key}: ${item.length || 0}`;
      return `${key}: ${summaryValue(item)}`;
    });
    return fields.join(' · ') || 'shared state';
  }
  const preferred = [
    'track_id', 'track_ids', 'hull_number', 'time_range', 'description', 'query',
    'registry_id', 'limit', 'after_start_time', 'after_track_id', 'gallery_images',
    'reference_ids', 'keyframe_ids', 'ship_segment_ids', 'registry_reference_ids',
  ];
  return preferred.filter((key) => key in args)
    .map((key) => `${key}: ${summaryValue(args[key])}`).join(' · ');
}

function completeToolEvent(event) {
  const callId = String(event.callId || '');
  const item = harnessToolCards.get(callId);
  if (!item) {
    createToolEvent({ ...event, type: 'tool_start', message: 'Tool call restored from memory' });
  }
  const target = harnessToolCards.get(callId);
  if (!target) return;
  // 失败的工具调用要一眼看得出来：事件带 status，早期事件只有 type
  const failed = event.type === 'error' || event.status === 'error';
  target.classList.remove('is-running');
  target.classList.add(failed ? 'is-error' : 'is-complete');
  const status = target.querySelector('.qa-tool-status');
  if (status) status.textContent = failed ? 'Failed' : 'Done';
  target.dataset.status = failed ? 'failed' : 'complete';
}

/* 证据工具一返回就刷新证据面板，不必等整轮结束——中间件保证收尾前一定会调用它。 */
function isEvidencePayload(result) {
  return !!result && typeof result === 'object' && ['shownKeyframeIds', 'shownShipSegmentIds', 'shownRegistryReferenceIds'].some((key) => Array.isArray(result[key]));
}

/* 子智能体行：一个 agent 只占一行，内容在该行内更新，结束后压成一行摘要。
   之前是每个流式块新建一行，一段推理能刷出十几行，轨迹没法看。 */
const agentLines = new Map();

function agentLine(agent) {
  const stream = document.getElementById('agentActivityStream');
  if (!stream) return null;
  const key = String(agent || '子智能体');
  const existing = agentLines.get(key);
  if (existing && existing.isConnected) return existing;
  stream.querySelector('.qa-empty-state')?.remove();
  const row = document.createElement('article');
  row.className = 'qa-agent-line';
  row.dataset.agent = key;
  row.dataset.state = 'running';
  row.innerHTML = `<span class="qa-agent-line-icon" aria-hidden="true">◈</span><span class="qa-agent-line-name">${escapeHtml(key)}</span><span class="qa-agent-line-label">思考 ·</span><span class="qa-agent-line-text"></span>`;
  stream.appendChild(row);
  agentLines.set(key, row);
  return row;
}

function agentLineSet(agent, text, options) {
  const opts = options || {};
  const row = agentLine(agent);
  if (!row) return;
  const body = row.querySelector('.qa-agent-line-text');
  if (body) {
    const next = opts.replace ? String(text || '') : `${body.textContent || ''}${text || ''}`;
    // 行内滚动：只保留尾部，长推理不会把这一行撑爆
    body.textContent = next.length > 400 ? next.slice(-400) : next;
  }
  if (opts.label) {
    const tag = row.querySelector('.qa-agent-line-label');
    if (tag) tag.textContent = `${opts.label} ·`;
  }
}

function agentLineNote(agent, note) {
  agentLineSet(agent, note ? `${note}\n` : '', { label: '工具' });
}

function finishAgentLines() {
  for (const row of agentLines.values()) {
    if (row.isConnected) row.dataset.state = 'done';
  }
}

function harnessErrorDetail(event) {
  return [...new Set([event.message, event.error, event.result?.error]
    .filter(Boolean).map((value) => String(value)))].join('\n');
}

function appendHarnessEvent(event) {
  if (!event || !event.type) return;
  const iteration = Number(event.iteration || event.round);
  if (Number.isFinite(iteration)) harnessIteration = Math.max(harnessIteration, iteration);
  if (event.type === 'skill') {
    appendSkillToInspector(event);
    updateSkillActivitySummary();
    if (event.agent) setAgentStatus(event.agent, 'running', '技能目录已就绪', iteration);
    setHarnessState(`Skills · ${harnessSkillNames.size} loaded`, 'running');
  } else if (event.type === 'tool_start') {
    const tool = String(event.tool || '');
    const internal = NON_BUSINESS_TOOLS.has(tool);
    const visible = tool !== 'task' && (!internal || tool === 'read_file');
    if (!internal && tool !== 'task') {
      harnessToolCount += 1;
      updateHarnessStats();
    }
    if (event.agent) setAgentStatus(event.agent, 'running', tool === 'read_file' ? '读取技能正文' : internal ? '更新共享状态' : `调用 ${event.label || tool}`, iteration);
    if (visible) createToolEvent(event);
    if (visible) setHarnessState(`${tool === 'read_file' ? 'Skill' : 'Tool'} · ${event.label || tool || 'running'}`, 'running');
  } else if (event.type === 'tool_result') {
    const tool = String(event.tool || '');
    const internal = NON_BUSINESS_TOOLS.has(tool);
    const visible = tool !== 'task' && (!internal || tool === 'read_file');
    const failed = event.status === 'error' || event.result?.ok === false;
    if (event.agent) {
      const action = tool === 'read_file'
        ? `技能正文${failed ? '读取失败' : '已读取'}`
        : internal ? '共享状态已更新' : `${event.label || tool} ${failed ? '失败' : '完成'}`;
      setAgentStatus(event.agent, failed ? 'failed' : 'running', action, iteration);
    }
    if (visible) {
      completeToolEvent(event);
      if (isEvidencePayload(event.result)) renderEvidence(event.result);
    }
    if (visible) setHarnessState(failed ? 'Tool failed' : 'Tool complete', 'running');
  } else if (event.type === 'model') {
    if (event.agent) {
      setAgentStatus(event.agent, 'running', '模型处理中', iteration);
    }
    setHarnessState('Running', 'running');
  } else if (event.type === 'agent_status') {
    setAgentStatus(event.agent, event.state, event.message, iteration);
    if (event.state === 'running') setHarnessState(`${AGENT_LABELS[event.agent] || event.agent} · ${event.message || 'Running'}`, 'running');
  } else if (event.type === 'status') {
    if (event.title === 'Agent 正在重连' && event.agent) {
      setAgentStatus(event.agent, 'reconnecting', event.message || '重连中', iteration);
    } else if (event.title !== '仍在执行' && !/已启动|开始工作/.test(String(event.title || ''))) {
      setHarnessState(event.message || event.title || 'Running', 'running');
    }
  } else if (event.type === 'complete') {
    finishAgentLines();
    for (const [agent, value] of harnessAgentStates) {
      if (value.state === 'running') setAgentStatus(agent, 'completed', '阶段完成', iteration);
    }
    harnessStageSummary = event.result?.state === 'cancelled' ? '问答已停止' : '问答完成';
    renderAgentStatus();
    setHarnessState(event.result?.state === 'cancelled' ? 'Stopped' : 'Complete', event.result?.state === 'cancelled' ? '' : 'complete');
  } else if (event.type === 'confirm') {
    renderAgentConfirmation(event);
    const agent = event.confirmation?.agent || 'executor';
    setAgentStatus(agent, 'waiting', '等待人工确认', iteration);
    setHarnessState('等待确认', 'awaiting_confirmation');
  } else if (event.type === 'error') {
    const detail = harnessErrorDetail(event);
    if (event.agent) setAgentStatus(event.agent, 'failed', detail || '阶段失败', iteration);
    else {
      for (const [agent, value] of harnessAgentStates) {
        if (value.state === 'running') setAgentStatus(agent, 'failed', detail || '阶段失败', iteration);
      }
    }
    appendStandardEvent(event, 'error', '!', 'ERROR', detail || 'Harness failed');
    setHarnessState('Failed', 'failed');
  }
  renderAgentStatus();
  scrollActivity();
  scrollConversation();
}

/* ---------------------------------------------------------------------------
 * 答案的 Markdown 渲染
 * 顺序很重要：**先转义再解析**。这样模型或文件名里的 < > & 只会是文本，
 * 不可能变成标签；代码块与行内代码用占位符取出，避免其中的 # - * 被当成 Markdown。
 * ------------------------------------------------------------------------- */
const MD_TOKEN = '\u0000';

function renderMarkdown(source) {
  const raw = String(source ?? '');
  if (!raw.trim()) return '';
  const codes = [];
  let text = raw.replace(/```[^\n]*\n?([\s\S]*?)```/g, (_match, body) => {
    codes.push(`<pre>${escapeHtml(body.replace(/\n$/, ''))}</pre>`);
    return `${MD_TOKEN}C${codes.length - 1}${MD_TOKEN}`;
  });
  text = escapeHtml(text);
  const inlines = [];
  text = text.replace(/`([^`\n]+)`/g, (_match, body) => {
    inlines.push(`<code>${body}</code>`);
    return `${MD_TOKEN}I${inlines.length - 1}${MD_TOKEN}`;
  });
  text = text
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  const lines = text.split('\n');
  const html = [];
  let list = null;
  const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) { closeList(); continue; }
    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed);
    if (heading) {
      closeList();
      const level = Math.min(6, heading[1].length + 1); // 正文 13px，所以 # 从 h2 起，别跳太大
      html.push(`<h${level}>${heading[2]}</h${level}>`);
      continue;
    }
    if (/^([-*_])\1{2,}$/.test(trimmed)) { closeList(); html.push('<hr>'); continue; }
    const bullet = /^[-*]\s+(.*)$/.exec(trimmed);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(trimmed);
    if (bullet || numbered) {
      const want = bullet ? 'ul' : 'ol';
      if (list !== want) { closeList(); html.push(`<${want}>`); list = want; }
      html.push(`<li>${(bullet || numbered)[1]}</li>`);
      continue;
    }
    if (/^&gt;\s?/.test(trimmed)) { closeList(); html.push(`<blockquote>${trimmed.replace(/^&gt;\s?/, '')}</blockquote>`); continue; }
    closeList();
    // 代码块已经是块级元素，不要再包一层 <p>
    if (trimmed.startsWith(`${MD_TOKEN}C`)) html.push(trimmed);
    else html.push(`<p>${trimmed}</p>`);
  }
  closeList();
  let out = html.join('');
  out = out.replace(new RegExp(`${MD_TOKEN}I(\\d+)${MD_TOKEN}`, 'g'), (_m, index) => inlines[Number(index)] || '');
  out = out.replace(new RegExp(`${MD_TOKEN}C(\\d+)${MD_TOKEN}`, 'g'), (_m, index) => codes[Number(index)] || '');
  return out;
}

function renderAgentAnswer(result) {
  const view = document.getElementById('agentFinalView');
  const answer = document.getElementById('agentAnswer');
  // 兜底：终局载荷必须是对象。曾经它会被服务端裁剪成字符串，
  // 前端于是显示「未生成回答 / 0 tool records」，看不出任何原因。
  const payload = result && typeof result === 'object' ? result : {};
  const malformed = result !== undefined && result !== null && typeof result !== 'object';
  const shared = payload.sharedState || payload.shared_state;
  if (shared) renderAgentPlan({ steps: (shared.plan || []).map((step) => ({ ...step, status: (shared.step_status || {})[step.id] || 'pending' })) });
  if (payload.state === 'awaiting_confirmation') {
    renderAgentConfirmation({ confirmation: payload.confirmation });
    setHarnessState('等待确认', 'awaiting_confirmation');
    return;
  }
  const cancelled = payload.state === 'cancelled';
  const stalled = payload.state === 'stalled';
  const awaitingInput = payload.state === 'awaiting_input';
  if (view) view.hidden = false;
  if (answer) {
    const text = payload.answerText || payload.answer;
    // 答案按 Markdown 渲染：模型输出的是 ## 结论 / **粗体** 这类结构，纯文本显示很难看。
    // renderMarkdown 内部先转义再解析，所以不会因为模型输出尖括号而注入标签。
    if (text) answer.innerHTML = renderMarkdown(text);
    else if (cancelled) answer.textContent = '本轮已停止。';
    else if (stalled) answer.textContent = '本轮未完成，未生成有效结论。';
    else if (malformed) answer.textContent = '本轮结果载荷异常，未能显示回答；请在「轨迹」页查看执行过程。';
    else answer.textContent = '未生成回答。';
  }
  renderEvidence(payload.evidence || null);
  if (cancelled) setHarnessState('Stopped', '');
  else if (stalled) setHarnessState('未完成', 'failed');
  else if (awaitingInput) setHarnessState('需要补充信息', '');
  else if (payload.partial && payload.success !== false) setHarnessState('已按现有证据收尾', 'complete');
  else setHarnessState(payload.success === false ? 'Failed' : 'Complete', payload.success === false ? 'failed' : 'complete');
  scrollActivity(true);
  scrollConversation(true);
}

function renderEvidence(evidence) {
  harnessEvidence = evidence || null;
  const node = document.getElementById('evidenceGallery');
  if (!node) return;
  const groups = [
    ['shownKeyframeIds', 'Keyframes', (id) => `<a class="evidence-card" href="/api/evidence/keyframes/${encodeURIComponent(id)}" target="_blank" rel="noopener"><img src="/api/evidence/keyframes/${encodeURIComponent(id)}" alt="Keyframe ${escapeHtml(id)}"><span>${escapeHtml(id)}</span></a>`],
    ['shownShipSegmentIds', 'Video clips', (id) => `<a class="evidence-card" href="/api/evidence/clips/${encodeURIComponent(id)}" target="_blank" rel="noopener"><img src="/api/evidence/clips/${encodeURIComponent(id)}/poster" alt="Video clip ${escapeHtml(id)}"><span>${escapeHtml(id)}</span></a>`],
    ['shownRegistryReferenceIds', 'Registry references', (id) => `<a class="evidence-card" href="/api/evidence/registry/${encodeURIComponent(id)}" target="_blank" rel="noopener"><img src="/api/evidence/registry/${encodeURIComponent(id)}" alt="Registry reference ${escapeHtml(id)}"><span>${escapeHtml(id)}</span></a>`],
  ];
  const maxItems = Math.max(0, Number(document.getElementById('evidenceMaxItems')?.value || 0));
  let shown = 0;
  const sections = groups.flatMap(([key, label, render]) => {
    const source = Array.isArray(evidence?.[key]) ? evidence[key] : [];
    const ids = maxItems ? source.slice(0, Math.max(0, maxItems - shown)) : source;
    shown += ids.length;
    return ids.length ? [`<section class="evidence-group"><strong>${label}</strong><div class="evidence-group-grid">${ids.map((id) => render(String(id))).join('')}</div></section>`] : [];
  });
  const total = groups.reduce((sum, [key]) => sum + (Array.isArray(evidence?.[key]) ? evidence[key].length : 0), 0);
  const count = document.getElementById('evidenceResultCount');
  if (count) count.textContent = total ? `${shown}/${total} item${total === 1 ? '' : 's'}` : 'No items';
  // 一行上下文：截图进论文时，这张图要能自己说清"这是哪一问的证据"。
  const context = document.getElementById('evidenceContext');
  if (context) {
    const asked = String(harnessResult?.question || currentSession?.title || '').trim();
    context.textContent = asked ? `Evidence for: ${asked}` : '';
    context.hidden = !asked;
  }
  node.style.setProperty('--qa-image-scale', String(Number(document.getElementById('evidenceImageScale')?.value || 0.5)));
  node.style.setProperty('--qa-video-scale', String(Number(document.getElementById('evidenceVideoScale')?.value || 0.25)));
  // 不再把整段原始 JSON 摊在面板里：换成能一眼看懂的要点。
  // 论文插图里那堆花括号只会占地方，真正要交代的是"有几条、什么 ID、用什么取的"。
  const tools = Array.isArray(evidence?.tools) ? [...new Set(evidence.tools.filter(Boolean))] : [];
  const facts = [
    ['Items', `${shown}/${total}`],
    ['Breakdown', groups.filter(([key]) => Array.isArray(evidence?.[key]) && evidence[key].length)
      .map(([key, label]) => `${label} ${evidence[key].length}`).join(' · ') || '—'],
    ['Display', String(evidence?.displayId || '—')],
    ['Retrieved by', tools.length ? tools.join(', ') : '—'],
  ];
  const meta = `<dl class="evidence-meta">${facts.map(([term, value]) =>
    `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>`;
  node.innerHTML = sections.length
    ? sections.join('') + meta
    : '<div class="qa-inspector-empty">No evidence available.</div>';
}

async function loadAgentMemorySummary(showNotice = false) {
  try {
    const response = await fetch('/api/agent/memory-summary');
    const data = await response.json();
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    const limit = document.getElementById('agentRoundLimit');
    if (limit) limit.textContent = `${data.trackCount ?? 0} tracks · ${data.keyframeCount ?? 0} keyframes`;
    if (showNotice && typeof showToast === 'function') showToast('Memory summary refreshed');
    return data;
  } catch (error) {
    if (showNotice && typeof showToast === 'function') showToast(error.message, 'error');
    return null;
  }
}

async function clearAgentMemory() {
  try {
    if ([...sessionRuns.values()].some((run) => run.status === 'running')) {
      throw new Error('请先停止进行中的问答，再清除记忆');
    }
    const response = await fetch('/api/agent/memory', {method: 'DELETE'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    sessionRuns.clear();
    startNewSession();
    await loadSessions();
    if (typeof showToast === 'function') showToast('QA memory cleared');
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

/* ── 一轮问答 ─────────────────────────────────────────────────────────────── */

/* 事件先入该轮的缓冲区，再决定要不要落到 DOM：正在看这个会话才渲染。 */
function dispatchRunEvent(run, event) {
  run.events.push(event);
  if (event.type === 'confirm') {
    run.status = 'awaiting_confirmation';
    run.result = event.result || run.result;
  }
  if (run.events.length > 500) run.events.splice(0, run.events.length - 500);
  if (currentSessionId !== run.id) return;
  if (event.type === 'confirm') syncComposer(run);
  appendHarnessEvent(event);
  // 轨迹页正开着就跟着长；没开着等切回去再重画
  if (qaView === 'trace') renderTrajectory();
}

async function streamAgentQuery(question, sessionId, run) {
  const response = await fetch('/api/agent/query/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, sessionId}),
    signal: run.controller?.signal,
  });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Request failed: ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;
  let errorEvent = null;
  const consume = (event) => {
    dispatchRunEvent(run, event);
    if (event.type === 'complete' || event.type === 'confirm') result = event.result || result;
    if (event.type === 'error') { errorEvent = event; result = event.result || result; }
  };
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) if (line.trim()) consume(JSON.parse(line));
    if (done) break;
  }
  if (buffer.trim()) consume(JSON.parse(buffer));
  if (errorEvent) {
    const error = new Error(errorEvent.message || errorEvent.result?.error || 'Harness failed');
    error.harnessEventRendered = true;
    error.result = result;
    throw error;
  }
  if (!result) throw new Error('No final result received');
  return result;
}

async function askAgent() {
  const question = document.getElementById('agentQuestion')?.value.trim();
  if (!question) { if (typeof showToast === 'function') showToast('请输入问题', 'error'); return; }
  const sessionId = currentSessionId || newSessionId();
  const status = sessionRuns.get(sessionId)?.status || currentSession?.turns?.at(-1)?.state;
  if (status === 'running' || status === 'awaiting_confirmation') {
    if (typeof showToast === 'function') showToast('请先完成当前问答或确认操作，也可以切换会话', 'error');
    return;
  }
  currentSessionId = sessionId;
  if (!currentSession || currentSession.sessionId !== sessionId) currentSession = {sessionId, title: question, turns: []};
  const run = {id: sessionId, question, events: [], result: null, error: null, status: 'running', stopping: false, controller: new AbortController()};
  sessionRuns.set(sessionId, run);
  // 发出去就清空输入框并收回到一行，保持输入区紧凑
  const input = document.getElementById('agentQuestion');
  if (input) { input.value = ''; autoGrowComposer(); }
  renderView();
  // 新会话立刻进列表并打上「回答中」标记：不等服务端把行写出来
  await loadSessions();

  try {
    const result = await streamAgentQuery(question, sessionId, run);
    run.result = result || null;
    run.status = result?.state === 'awaiting_confirmation' ? 'awaiting_confirmation' : result?.state === 'cancelled' ? 'stopped' : result?.success === false ? 'error' : 'done';
  } catch (error) {
    // 用户按过停止：无论流是抛 AbortError 还是直接断掉，都算「已停止」而不是失败
    if (error.name === 'AbortError' || run.stopping) {
      run.status = 'stopped';
      run.error = '已停止';
    } else {
      run.status = 'error';
      run.result = error.result || run.result;
      run.error = error.message;
      if (!error.harnessEventRendered && currentSessionId === sessionId) {
        appendHarnessEvent({type: 'error', title: 'Harness 执行失败', message: error.message});
      }
      if (typeof showToast === 'function') showToast(error.message, 'error');
    }
  } finally {
    run.controller = null;
    if (currentSessionId === sessionId) renderView();
    await loadSessions();
    try {
      const detail = await fetchSession(sessionId);
      if (currentSessionId === sessionId) {
        currentSession = detail;
        renderView();
      }
    } catch (_error) {
      /* 存档刷新失败不影响已经渲染好的本轮结果 */
    }
  }
}

async function stopAgentRun() {
  const run = currentSessionId ? sessionRuns.get(currentSessionId) : null;
  if (!run || run.status !== 'running') return;
  run.stopping = true;   // 先记意图：随后流无论是报错还是直接断，都按「已停止」处理
  const stop = document.getElementById('btnStopAgent');
  if (stop) stop.disabled = true;
  setHarnessState('Stopping…', 'running');
  try {
    // 先让服务端在当前步骤收尾，再断开前端这条流，避免它继续等到超时
    await fetch(`/api/agent/sessions/${encodeURIComponent(run.id)}/stop`, {method: 'POST'});
  } catch (_error) {
    /* 服务端没收到也要能把前端停下来 */
  }
  run.controller?.abort();
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('agentQuestion');
  input?.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); askAgent(); }
  });
  input?.addEventListener('input', autoGrowComposer);
  autoGrowComposer();
  for (const id of ['evidenceImageScale', 'evidenceVideoScale', 'evidenceMaxItems']) {
    const control = document.getElementById(id);
    const output = document.getElementById(`${id}Value`);
    control?.addEventListener('input', () => {
      if (output) output.textContent = id === 'evidenceMaxItems' ? (Number(control.value) ? control.value : 'All') : `${Math.round(Number(control.value) * 100)}%`;
      if (harnessEvidence) renderEvidence(harnessEvidence);
    });
  }
  bindConfirmationButtons();
  loadAgentMemorySummary();
  loadSessions();
});

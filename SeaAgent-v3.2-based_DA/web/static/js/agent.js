/* Sea-Video-Harness QA interaction layer. Public activity only; no hidden chain-of-thought is rendered. */

/* ── 会话与运行状态 ───────────────────────────────────────────────────────────
   一个会话一轮问答。sessionRuns 以 session_id 为键保存「进行中或刚结束」的那一轮：
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
let harnessEventCount = 0;
let harnessToolCount = 0;
const harnessToolCards = new Map();
const harnessSkillNames = new Set();
let harnessSkillActivityCard = null;

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
  const eventNode = document.getElementById('qaEventCount');
  const toolNode = document.getElementById('qaToolCount');
  const skillNode = document.getElementById('qaSkillCount');
  if (eventNode) eventNode.textContent = String(harnessEventCount);
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
  const summary = document.getElementById('agentToolSummary');
  if (summary) summary.textContent = 'Waiting for completion';
  harnessEventCount = 0;
  harnessToolCount = 0;
  harnessToolCards.clear();
  harnessSkillNames.clear();
  harnessSkillActivityCard = null;
  harnessResult = null;
  harnessEvidence = null;
  updateHarnessStats();
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
  if (button) {
    button.disabled = running;
    button.title = running ? '本轮进行中…' : '运行（Ctrl / ⌘ + Enter）';
  }
  if (stop) {
    stop.hidden = !running;
    stop.disabled = !running;
  }
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
  if (turn?.state === 'stalled') return '（这一轮工具调用连续失败，已按现有结果收尾）';
  return turn?.state || '（该轮未留下回答）';
}

/* 把某一轮问答还原到主面板：回答、工具名序列与证据都来自会话存档。 */
function renderRestoredTurn(turn) {
  const view = document.getElementById('agentFinalView');
  const answer = document.getElementById('agentAnswer');
  if (view) view.hidden = false;
  if (answer) answer.textContent = turnAnswer(turn) || '（该轮未留下回答）';
  const summary = document.getElementById('agentToolSummary');
  const tools = document.getElementById('agentResultTools');
  const chain = Array.isArray(turn?.toolChain) ? turn.toolChain : [];
  if (summary) summary.textContent = chain.length ? `${chain.length} tool call${chain.length === 1 ? '' : 's'} · from session memory` : 'No tool records.';
  if (tools) {
    tools.innerHTML = chain.length
      ? chain.map((name) => `<article class="qa-trace-record"><strong>${escapeHtml(name)}</strong><span>completed</span></article>`).join('')
      : '<div class="qa-inspector-empty">这轮没有留下工具调用记录。</div>';
  }
  renderEvidence(turn?.evidence || null);
  const state = turn?.state === 'error' ? 'Failed' : turn?.state === 'cancelled' ? 'Stopped' : 'Restored';
  setHarnessState(state, turn?.state === 'error' ? 'failed' : turn?.state === 'cancelled' ? '' : 'complete');
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
  const tool = String(event.label || event.tool || 'Tool');
  const item = document.createElement('article');
  item.className = 'qa-tool-event is-running';
  item.dataset.callId = callId;
  item.innerHTML = `<div class="qa-tool-event-head"><span class="qa-event-icon" aria-hidden="true">↗</span><div class="qa-event-main"><div class="qa-event-title"><span>${escapeHtml(tool)}</span><em class="qa-event-label">TOOL</em></div><div class="qa-event-message">${escapeHtml(event.message || 'Tool call submitted')}</div></div><span class="qa-tool-status">Running</span></div><details><summary>View input</summary><pre>${escapeHtml(compact(event.arguments || {}, 1800))}</pre></details>`;
  stream.appendChild(item);
  harnessToolCards.set(callId, item);
  harnessToolCount += 1;
  return item;
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
  if (status) status.textContent = failed ? 'Failed' : 'Complete';
  const message = target.querySelector('.qa-event-message');
  if (message) message.textContent = event.message || 'Tool call complete';
  const details = target.querySelector('details');
  if (details && event.result !== undefined) {
    details.querySelector('summary').textContent = 'View output';
    const pre = details.querySelector('pre');
    if (pre) pre.textContent = compact(event.result, 1800);
  }
}

/* 证据工具一返回就刷新证据面板，不必等整轮结束——中间件保证收尾前一定会调用它。 */
function isEvidencePayload(result) {
  return !!result && typeof result === 'object' && ['shownKeyframeIds', 'shownShipSegmentIds', 'shownRegistryReferenceIds'].some((key) => Array.isArray(result[key]));
}

function appendHarnessEvent(event) {
  if (!event || !event.type) return;
  harnessEventCount += 1;
  updateHarnessStats();
  if (event.type === 'skill') {
    appendSkillToInspector(event);
    updateSkillActivitySummary();
    setHarnessState(`Skills · ${harnessSkillNames.size} loaded`, 'running');
  } else if (event.type === 'tool_start') {
    createToolEvent(event);
    setHarnessState(`Tool · ${event.label || event.tool || 'running'}`, 'running');
  } else if (event.type === 'tool_result') {
    completeToolEvent(event);
    if (isEvidencePayload(event.result)) renderEvidence(event.result);
    setHarnessState('Tool complete', 'running');
  } else if (event.type === 'model') {
    appendStandardEvent(event, 'model', '◌', 'MODEL', event.message || 'Public model step updated');
    setHarnessState('Model response', 'running');
  } else if (event.type === 'status') {
    appendStandardEvent(event, 'status', '◈', 'SYSTEM', event.message || 'Harness ready');
  } else if (event.type === 'complete') {
    appendStandardEvent(event, 'complete', '✓', 'DONE', event.message || 'Answer and evidence generated');
    setHarnessState(event.result?.state === 'cancelled' ? 'Stopped' : 'Complete', event.result?.state === 'cancelled' ? '' : 'complete');
  } else if (event.type === 'error') {
    const detail = [event.message, event.error, event.result?.error].filter(Boolean).join('\n');
    appendStandardEvent(event, 'error', '!', 'ERROR', detail || 'Harness failed');
    setHarnessState('Failed', 'failed');
  }
  scrollActivity();
  scrollConversation();
}

function renderToolRecords(records) {
  const node = document.getElementById('agentResultTools');
  const summary = document.getElementById('agentToolSummary');
  if (!node) return;
  const items = Array.isArray(records) ? records : [];
  if (summary) summary.textContent = `${items.length} tool record${items.length === 1 ? '' : 's'}`;
  node.innerHTML = items.length ? items.map((item) => `<article class="qa-trace-record"><strong>${escapeHtml(item.label || item.tool || 'tool')}</strong><code>${escapeHtml(compact(item.result || item.arguments || {}, 220))}</code><span>${escapeHtml(item.status || '')}</span></article>`).join('') : '<div class="qa-inspector-empty">No tool records.</div>';
}

function renderAgentAnswer(result) {
  const view = document.getElementById('agentFinalView');
  const answer = document.getElementById('agentAnswer');
  const cancelled = result?.state === 'cancelled';
  const stalled = result?.state === 'stalled';
  if (view) view.hidden = false;
  if (answer) answer.textContent = result?.answerText || result?.answer || (cancelled ? '本轮已停止。' : stalled ? '本轮工具调用连续失败，已按现有结果收尾。' : '未生成回答。');
  renderToolRecords(result?.toolRecords || result?.tool_records);
  renderEvidence(result?.evidence || null);
  if (cancelled) setHarnessState('Stopped', '');
  else if (stalled) setHarnessState('已收尾', 'complete');
  else setHarnessState(result?.success === false ? 'Failed' : 'Complete', result?.success === false ? 'failed' : 'complete');
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
  node.style.setProperty('--qa-image-scale', String(Number(document.getElementById('evidenceImageScale')?.value || 0.5)));
  node.style.setProperty('--qa-video-scale', String(Number(document.getElementById('evidenceVideoScale')?.value || 0.25)));
  node.innerHTML = sections.length ? sections.join('') + `<pre class="evidence-json">${escapeHtml(compact(evidence, 5000))}</pre>` : '<div class="qa-inspector-empty">No evidence available.</div>';
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
    // 先停掉所有在跑的会话，否则它们的轮次会在清空后又写回记忆
    for (const run of sessionRuns.values()) run.controller?.abort();
    sessionRuns.clear();
    const response = await fetch('/api/agent/memory', {method: 'DELETE'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
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
  if (run.events.length > 500) run.events.splice(0, run.events.length - 500);
  if (currentSessionId === run.id) appendHarnessEvent(event);
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
    if (event.type === 'complete') result = event.result || result;
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
  if (sessionRuns.get(sessionId)?.status === 'running') {
    if (typeof showToast === 'function') showToast('这个会话还在回答，先停止或换一个会话', 'error');
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
    run.status = result?.state === 'cancelled' ? 'stopped' : result?.success === false ? 'error' : 'done';
  } catch (error) {
    // 用户按过停止：无论流是抛 AbortError 还是直接断掉，都算「已停止」而不是失败
    if (error.name === 'AbortError' || run.stopping) {
      run.status = 'stopped';
      run.error = '已停止';
    } else {
      run.status = 'error';
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
  loadAgentMemorySummary();
  loadSessions();
});

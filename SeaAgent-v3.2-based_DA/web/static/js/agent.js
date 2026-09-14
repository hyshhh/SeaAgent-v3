/* Sea-Video-Harness QA interaction layer. Public activity only; no hidden chain-of-thought is rendered. */
let harnessResult = null;
let harnessEvidence = null;
let harnessEventCount = 0;
let harnessToolCount = 0;
const harnessToolCards = new Map();
const harnessSkillNames = new Set();
let harnessSkillActivityCard = null;
/* 会话状态：currentSessionId 为 null 表示下一轮问答会开一段新会话，否则在该会话里追问。 */
let currentSessionId = null;
let currentSession = null;
let sessionSummaries = [];

function useQuestion(text) {
  const input = document.getElementById('agentQuestion');
  if (input) {
    input.value = text;
    input.focus();
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

/* 会话时间戳由后端给 ISO-8601（带时区），这里统一收成 MM-DD HH:mm。 */
function formatSessionTime(value) {
  const date = new Date(String(value || ''));
  if (Number.isNaN(date.getTime())) return '—';
  const pad = (number) => String(number).padStart(2, '0');
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
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

function resetThoughtStream() {
  const stream = document.getElementById('agentActivityStream');
  if (stream) stream.innerHTML = '<div class="qa-empty-state">Activity will appear here when the harness starts.</div>';
  const final = document.getElementById('agentFinalView');
  if (final) final.hidden = true;
  const welcome = document.getElementById('qaWelcome');
  if (welcome) welcome.hidden = true;
  const userTurn = document.getElementById('qaUserTurn');
  if (userTurn) userTurn.hidden = false;
  const skills = document.getElementById('qaSkillList');
  if (skills) skills.innerHTML = '<div class="qa-inspector-empty">Skills load with the harness.</div>';
  const evidence = document.getElementById('evidenceGallery');
  if (evidence) evidence.innerHTML = '<div class="qa-inspector-empty">No evidence available.</div>';
  const count = document.getElementById('evidenceResultCount');
  if (count) count.textContent = 'Waiting';
  harnessEventCount = 0;
  harnessToolCount = 0;
  harnessToolCards.clear();
  harnessSkillNames.clear();
  harnessSkillActivityCard = null;
  harnessResult = null;
  harnessEvidence = null;
  updateHarnessStats();
  setHarnessState('Running', 'running');
}

/* ---------------------------------------------------------------- 会话记忆 */

function renderSessionList() {
  const list = document.getElementById('qaSessionList');
  if (!list) return;
  if (!sessionSummaries.length) {
    list.innerHTML = '<div class="qa-inspector-empty">暂无会话记录</div>';
    return;
  }
  list.innerHTML = sessionSummaries.map((session) => {
    const active = session.sessionId === currentSessionId ? ' is-active' : '';
    const turns = Number(session.turnCount || 0);
    return `<div class="qa-session-item${active}" role="button" tabindex="0" data-session-id="${escapeHtml(session.sessionId)}" onclick="openSession(this.dataset.sessionId)" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openSession(this.dataset.sessionId);}">
      <strong title="${escapeHtml(session.title || '未命名会话')}">${escapeHtml(session.title || '未命名会话')}</strong>
      <small>${escapeHtml(formatSessionTime(session.updatedAt))} · ${turns} 轮</small>
      <button type="button" class="qa-session-delete" title="删除该会话" onclick="event.stopPropagation();deleteSession(this.closest('.qa-session-item').dataset.sessionId)">✕</button>
    </div>`;
  }).join('');
}

async function loadSessions() {
  try {
    const response = await fetch('/api/agent/sessions');
    const data = await response.json();
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    sessionSummaries = Array.isArray(data?.sessions) ? data.sessions : [];
  } catch (_error) {
    sessionSummaries = [];
  }
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
function renderSessionHistory(session) {
  const node = document.getElementById('qaSessionHistory');
  if (!node) return;
  const turns = Array.isArray(session?.turns) ? session.turns : [];
  const past = turns.slice(0, -1);
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
      <p class="qa-history-answer">${escapeHtml(compact(turn.answer || turn.state || '（该轮未留下回答）', 420))}</p>
    </article>`).join('');
  node.hidden = false;
}

/* 把某一轮问答还原到主面板：回答、工具名序列与证据都来自会话存档。 */
function renderRestoredTurn(turn) {
  const view = document.getElementById('agentFinalView');
  const answer = document.getElementById('agentAnswer');
  if (view) view.hidden = false;
  if (answer) answer.textContent = turn?.answer || '（该轮未留下回答）';
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
  setHarnessState(turn?.state === 'error' ? 'Failed' : 'Restored', turn?.state === 'error' ? 'failed' : 'complete');
}

async function openSession(sessionId) {
  if (!sessionId) return;
  try {
    const session = await fetchSession(sessionId);
    currentSession = session;
    currentSessionId = session.sessionId;
    renderSessionList();
    resetThoughtStream();
    const turns = Array.isArray(session.turns) ? session.turns : [];
    const last = turns[turns.length - 1] || null;
    renderRestoredTurn(last);
    renderSessionHistory(session);
    const questionNode = document.getElementById('qaUserQuestion');
    if (questionNode) questionNode.textContent = last?.question || '';
    const welcome = document.getElementById('qaWelcome');
    if (welcome) welcome.hidden = true;
    const userTurn = document.getElementById('qaUserTurn');
    if (userTurn) userTurn.hidden = false;
    scrollConversation(true);
    scrollActivity(true);
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

function startNewSession() {
  currentSessionId = null;
  currentSession = null;
  renderSessionList();
  resetThoughtStream();
  const history = document.getElementById('qaSessionHistory');
  if (history) { history.hidden = true; history.innerHTML = ''; }
  const welcome = document.getElementById('qaWelcome');
  if (welcome) welcome.hidden = false;
  const userTurn = document.getElementById('qaUserTurn');
  if (userTurn) userTurn.hidden = true;
  const answer = document.getElementById('agentAnswer');
  if (answer) answer.textContent = 'The final answer will appear here after execution.';
  const tools = document.getElementById('agentResultTools');
  if (tools) tools.innerHTML = '<div class="qa-inspector-empty">No tool records.</div>';
  setHarnessState('Ready', '');
}

async function deleteSession(sessionId) {
  if (!sessionId) return;
  try {
    const response = await fetch(`/api/agent/sessions/${encodeURIComponent(sessionId)}`, {method: 'DELETE'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    if (currentSessionId === sessionId) startNewSession();
    await loadSessions();
    if (typeof showToast === 'function') showToast('会话已删除');
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

/* 一轮问答结束后刷新会话栏，并把刚结束的那轮挤出历史区（主面板已经在展示它）。 */
async function refreshCurrentSession() {
  await loadSessions();
  if (currentSessionId) {
    try {
      currentSession = await fetchSession(currentSessionId);
      renderSessionHistory(currentSession);
    } catch (_error) {
      /* 会话栏刷新失败不影响已经渲染好的本轮回答 */
    }
  }
  renderSessionList();
}

/* ---------------------------------------------------------------- 事件渲染 */

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
  target.classList.remove('is-running');
  target.classList.add(event.type === 'error' ? 'is-error' : 'is-complete');
  const status = target.querySelector('.qa-tool-status');
  if (status) status.textContent = event.type === 'error' ? 'Failed' : 'Complete';
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
    setHarnessState('Complete', 'complete');
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
  if (view) view.hidden = false;
  if (answer) answer.textContent = result?.answerText || result?.answer || '未生成回答。';
  renderToolRecords(result?.toolRecords || result?.tool_records);
  renderEvidence(result?.evidence || null);
  setHarnessState(result?.success === false ? 'Failed' : 'Complete', result?.success === false ? 'failed' : 'complete');
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
    const response = await fetch('/api/agent/memory', {method: 'DELETE'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data?.detail || `Request failed: ${response.status}`);
    startNewSession();
    await loadSessions();
    setHarnessState('Ready', '');
    if (typeof showToast === 'function') showToast('QA memory cleared');
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

async function streamAgentQuery(question, sessionId) {
  const payload = {question};
  if (sessionId) payload.sessionId = sessionId;
  const response = await fetch('/api/agent/query/stream', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Request failed: ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;
  let errorEvent = null;
  const consume = (event) => {
    appendHarnessEvent(event);
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
  const button = document.getElementById('btnAskAgent');
  if (!question) return typeof showToast === 'function' && showToast('请输入问题', 'error');
  resetThoughtStream();
  const questionNode = document.getElementById('qaUserQuestion');
  if (questionNode) questionNode.textContent = question;
  if (button) { button.disabled = true; button.querySelector('span')?.replaceChildren(document.createTextNode('Running…')); }
  setHarnessState('Running', 'running');
  try {
    harnessResult = await streamAgentQuery(question, currentSessionId);
    renderAgentAnswer(harnessResult);
  } catch (error) {
    if (!error.harnessEventRendered) appendHarnessEvent({type: 'error', title: 'Harness 执行失败', message: error.message});
    if (error.result) { harnessResult = error.result; renderAgentAnswer(error.result); }
    if (typeof showToast === 'function') showToast(error.message, 'error');
  } finally {
    if (button) { button.disabled = false; button.querySelector('span')?.replaceChildren(document.createTextNode('Run Harness')); }
    // 后端每轮都会把问答写进会话，拿到 sessionId 后即成为当前会话，后续提问就是追问
    if (harnessResult?.sessionId) currentSessionId = harnessResult.sessionId;
    await refreshCurrentSession();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('agentQuestion');
  input?.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); askAgent(); }
  });
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

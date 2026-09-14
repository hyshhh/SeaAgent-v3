/* Sea-Video-Harness QA interaction layer. Public activity only; no hidden chain-of-thought is rendered. */
let harnessResult = null;
let harnessEvidence = null;
let harnessEventCount = 0;
let harnessToolCount = 0;
const harnessToolCards = new Map();
const harnessSkillNames = new Set();

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
  harnessResult = null;
  harnessEvidence = null;
  updateHarnessStats();
  setHarnessState('Running', 'running');
}

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

function appendHarnessEvent(event) {
  if (!event || !event.type) return;
  harnessEventCount += 1;
  updateHarnessStats();
  if (event.type === 'skill') {
    appendSkillToInspector(event);
    appendStandardEvent(event, 'skill', '✦', 'SKILL', event.message || 'Skill loaded');
    setHarnessState(`Skill · ${event.skill || event.name || 'loaded'}`, 'running');
  } else if (event.type === 'tool_start') {
    createToolEvent(event);
    setHarnessState(`Tool · ${event.label || event.tool || 'running'}`, 'running');
  } else if (event.type === 'tool_result') {
    completeToolEvent(event);
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
    resetThoughtStream();
    const welcome = document.getElementById('qaWelcome');
    const userTurn = document.getElementById('qaUserTurn');
    if (welcome) welcome.hidden = false;
    if (userTurn) userTurn.hidden = true;
    setHarnessState('Ready', '');
    if (typeof showToast === 'function') showToast('QA memory cleared');
  } catch (error) {
    if (typeof showToast === 'function') showToast(error.message, 'error');
  }
}

async function streamAgentQuery(question) {
  const response = await fetch('/api/agent/query/stream', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question})});
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
    harnessResult = await streamAgentQuery(question);
    renderAgentAnswer(harnessResult);
  } catch (error) {
    if (!error.harnessEventRendered) appendHarnessEvent({type: 'error', title: 'Harness 执行失败', message: error.message});
    if (error.result) { harnessResult = error.result; renderAgentAnswer(error.result); }
    if (typeof showToast === 'function') showToast(error.message, 'error');
  } finally {
    if (button) { button.disabled = false; button.querySelector('span')?.replaceChildren(document.createTextNode('Run Harness')); }
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
});

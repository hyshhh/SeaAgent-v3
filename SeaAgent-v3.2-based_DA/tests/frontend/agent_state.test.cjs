const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../web/static/js/agent.js'), 'utf8');

function page(fetch = async () => ({ ok: true, json: async () => ({ sessions: [] }) })) {
  const elements = new Map();
  for (const id of [
    'agentPlanPanel', 'agentPlanList', 'agentPlanProgress', 'agentPlanMeterFill',
    'agentConfirmPanel', 'agentConfirmBody', 'agentConfirmHint', 'agentConfirmFeedback',
    'agentFinalView', 'agentAnswer', 'btnAskAgent', 'btnStopAgent',
    'agentThinkingState',
  ]) {
    elements.set(id, {
      hidden: true, innerHTML: '', textContent: '', value: '', disabled: false,
      dataset: {}, style: { setProperty() {} },
    });
  }
  const context = vm.createContext({
    document: { getElementById: (id) => elements.get(id) || null, addEventListener() {} },
    window: {}, requestAnimationFrame: (fn) => fn(), fetch, TextDecoder, Uint8Array,
    AbortController, console,
  });
  vm.runInContext(source, context);
  return { elements, run: (code) => vm.runInContext(code, context) };
}

const confirmation = { actions: [{ tool: 'add_registry_vessel', arguments: { hull_number: '003' } }] };

test('shared plans render progress, escape content and clear when switching sessions', () => {
  const ui = page();
  ui.run(`renderAgentPlan({ steps: [
    { content: '<img src=x onerror=alert(1)>', status: 'completed' },
    { content: 'verify', status: 'pending' }
  ] });`);
  assert.equal(ui.elements.get('agentPlanPanel').hidden, false);
  assert.match(ui.elements.get('agentPlanList').innerHTML, /&lt;img/);
  assert.equal(ui.elements.get('agentPlanMeterFill').style.width, '50%');
  ui.run('resetActivityDom()');
  assert.equal(ui.elements.get('agentPlanPanel').hidden, true);
  assert.equal(ui.elements.get('agentPlanList').innerHTML, '');
  assert.equal(ui.elements.get('agentConfirmPanel').hidden, true);
});

test('a split NDJSON confirmation is accepted as a terminal result', async () => {
  const result = { success: false, state: 'awaiting_confirmation', confirmation };
  const line = JSON.stringify({ type: 'confirm', confirmation, result }) + '\n';
  const chunks = [line.slice(0, 20), line.slice(20)].map((text) => new TextEncoder().encode(text));
  const ui = page(async () => ({
    ok: true, body: { getReader: () => ({
      read: async () => chunks.length ? { value: chunks.shift(), done: false } : { done: true },
    }) },
  }));
  ui.run("currentSessionId = 't'");
  const returned = await ui.run("streamAgentQuery('q', 't', { id: 't', events: [] })");
  assert.equal(returned.state, 'awaiting_confirmation');
  assert.equal(ui.elements.get('agentConfirmPanel').hidden, false);
  assert.equal(ui.run('pendingConfirmation.sessionId'), 't');
});

test('confirming twice sends one request and displays the actual returned answer', async () => {
  let release;
  let requests = 0;
  const result = { success: true, state: 'completed', answerText: 'verified',
    sharedState: { plan: [{ id: 'scan', content: 'scan' }], step_status: { scan: 'completed' } } };
  const ui = page(async (url) => {
    if (url === '/api/agent/query/resume') {
      requests += 1;
      await new Promise((resolve) => { release = resolve; });
      return { ok: true, json: async () => result };
    }
    return { ok: true, json: async () => ({ sessions: [] }) };
  });
  ui.run(`currentSessionId = 't';
    sessionRuns.set('t', { id: 't', status: 'awaiting_confirmation', events: [{type: 'confirm'}] });
    renderAgentConfirmation({ confirmation: ${JSON.stringify(confirmation)} });`);
  const pending = ui.run("submitConfirmation('approve')");
  await ui.run("submitConfirmation('approve')");
  assert.equal(requests, 1);
  release();
  await pending;
  assert.equal(ui.run("sessionRuns.get('t').status"), 'done');
  assert.equal(ui.run("sessionRuns.get('t').events.length"), 0);
  assert.match(ui.elements.get('agentAnswer').innerHTML, /verified/);
  assert.equal(ui.elements.get('agentPlanMeterFill').style.width, '100%');
  assert.equal(ui.elements.get('agentConfirmPanel').hidden, true);
  assert.equal(ui.elements.get('btnAskAgent').disabled, false);
});

test('a confirmation response cannot replace a newly selected session', async () => {
  let release;
  const ui = page(async () => {
    await new Promise((resolve) => { release = resolve; });
    return { ok: false, status: 503 };
  });
  ui.run(`currentSessionId = 'a';
    renderAgentConfirmation({ confirmation: ${JSON.stringify(confirmation)} });`);
  const pending = ui.run("submitConfirmation('approve')");
  ui.run(`currentSessionId = 'b'; resetActivityDom();
    renderAgentConfirmation({ confirmation: ${JSON.stringify(confirmation)} });`);
  release();
  await pending;
  assert.equal(ui.elements.get('agentConfirmPanel').dataset.state, 'pending');
  assert.equal(ui.run('pendingConfirmation.sessionId'), 'b');
  assert.equal(ui.run("sessionRuns.get('a').status"), 'awaiting_confirmation');
});

test('history restores a pending confirmation and completed shared plan', () => {
  const ui = page();
  ui.run(`currentSessionId = 't'; renderRestoredTurn({
    state: 'awaiting_confirmation', confirmation: ${JSON.stringify(confirmation)},
    plan: { steps: [{ id: 'scan', content: 'scan', status: 'completed' }] }
  });`);
  assert.equal(ui.elements.get('agentPlanMeterFill').style.width, '100%');
  assert.equal(ui.elements.get('agentConfirmPanel').hidden, false);
  assert.equal(ui.elements.get('btnAskAgent').disabled, true);
});

test('failed memory cleanup preserves local session records', async () => {
  const ui = page(async () => ({ ok: false, status: 409, json: async () => ({ detail: 'busy' }) }));
  ui.run("sessionRuns.set('t', {id: 't', status: 'done', events: []})");
  await ui.run('clearAgentMemory()');
  assert.equal(ui.run("sessionRuns.has('t')"), true);
});

test('a converged answer is labeled partial and preserves unfinished steps', () => {
  const ui = page();
  ui.run(`renderAgentAnswer({ success: true, state: 'completed', partial: true,
    answerText: '已取得 kf1，尚未验证目标',
    sharedState: { plan: [{ id: 'verify', content: 'verify target' }],
      step_status: { verify: 'pending' } } });`);
  assert.equal(ui.elements.get('agentThinkingState').textContent, '已按现有证据收尾');
  assert.equal(ui.elements.get('agentPlanMeterFill').style.width, '0%');
  assert.match(ui.elements.get('agentAnswer').innerHTML, /kf1/);
  ui.run(`renderRestoredTurn({ state: 'completed', partial: true,
    answer: '已取得 kf1，尚未验证目标' });`);
  assert.equal(ui.elements.get('agentThinkingState').textContent, '已按现有证据收尾');
});

test('a stalled run is labeled incomplete instead of completed', () => {
  const ui = page();
  ui.run(`renderAgentAnswer({ state: 'stalled', answerText: 'Planner 未提交状态' });`);
  assert.match(ui.elements.get('agentAnswer').innerHTML, /Planner 未提交状态/);
  assert.equal(ui.elements.get('agentThinkingState').textContent, '未完成');
  ui.run(`renderRestoredTurn({ state: 'stalled', answer: 'Planner 未提交状态' });`);
  assert.equal(ui.elements.get('agentThinkingState').textContent, '未完成');
});

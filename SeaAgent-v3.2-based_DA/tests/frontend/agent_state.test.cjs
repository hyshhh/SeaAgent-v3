const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../web/static/js/agent.js'), 'utf8');
const template = fs.readFileSync(path.join(__dirname, '../../web/templates/index.html'), 'utf8');

function page(fetch = async () => ({ ok: true, json: async () => ({ sessions: [] }) })) {
  const elements = new Map();
  const activityRows = [];
  const activityStream = {
    innerHTML: '',
    querySelector: () => null,
    appendChild(row) { row.parentElement = this; row.isConnected = true; activityRows.push(row); },
  };
  for (const id of [
    'agentPlanPanel', 'agentPlanList', 'agentPlanProgress', 'agentPlanMeterFill',
    'agentConfirmPanel', 'agentConfirmBody', 'agentConfirmHint', 'agentConfirmFeedback',
    'qaChatConfirmationMount', 'qaTraceConfirmationMount',
    'btnConfirmApprove', 'btnConfirmReject',
    'agentFinalView', 'agentAnswer', 'btnAskAgent', 'btnStopAgent',
    'agentThinkingState',
    'qaSharedStateGroups', 'qaSharedStateMeta',
    'evidenceGallery',
  ]) {
    elements.set(id, {
      hidden: true, innerHTML: '', textContent: '', value: '', disabled: false,
      dataset: {}, style: { setProperty() {} }, appendChild(child) { child.parentElement = this; },
    });
  }
  const context = vm.createContext({
    document: {
      getElementById: (id) => id === 'agentActivityStream' ? activityStream : elements.get(id) || null,
      createElement: () => {
        const message = { textContent: '' };
        return {
          dataset: {}, isConnected: false, innerHTML: '',
          querySelector: (selector) => selector === '.qa-event-message' ? message : null,
          message,
        };
      },
      addEventListener() {},
    },
    window: {}, requestAnimationFrame: (fn) => fn(), fetch, TextDecoder, Uint8Array,
    AbortController, console,
  });
  vm.runInContext(source, context);
  return { elements, activityRows, run: (code) => vm.runInContext(code, context) };
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

test('shared state renders escaped fields and its snapshot version', () => {
  const ui = page();
  ui.run(`renderSharedState({ version: 4, requirements: ['<script>'], decision: 'continue' });`);
  assert.match(ui.elements.get('qaSharedStateGroups').innerHTML, /&lt;script&gt;/);
  assert.match(ui.elements.get('qaSharedStateGroups').innerHTML, /Requirements/);
  assert.equal(ui.elements.get('qaSharedStateMeta').textContent, 'Version 4');
});

test('evidence statistics render even when the query returns no visual items', () => {
  const ui = page();
  ui.run(`renderEvidence({ statistics: {
    queryByTool: [{ tool: 'count_total_entities', statistics: { count: 0, entityCount: 0, countConfidence: 'confirmed' } }],
    display: { trackCount: 0, keyframeCount: 0, clipCount: 0, registryReferenceCount: 0 }
  } });`);
  const html = ui.elements.get('evidenceGallery').innerHTML;
  assert.match(html, /No evidence available/);
  assert.match(html, /Count: 0/);
  assert.match(html, /Entities: 0/);
  assert.match(html, /Evidence display/);
});

test('skill loading and reading events collapse into one summary row per agent and phase', () => {
  const ui = page();
  ui.run(`
    updateSkillActivitySummary('executor', 'load', 'query', 'complete');
    updateSkillActivitySummary('executor', 'load', 'finalize', 'complete');
    updateSkillActivitySummary('executor', 'read', 'skills/query/SKILL.md');
    updateSkillActivitySummary('executor', 'read', '', 'complete');
  `);
  assert.equal(ui.activityRows.length, 2);
  assert.match(ui.activityRows[0].innerHTML, /Executor · Skill loading/);
  assert.match(ui.activityRows[0].message.textContent, /2 skills · query, finalize/);
  assert.match(ui.activityRows[1].innerHTML, /Executor · Skill reading/);
  assert.match(ui.activityRows[1].message.textContent, /Read 1 skill file/);
});

test('trajectory omits skill filesystem tools and keeps business tools', () => {
  const ui = page();
  const steps = ui.run(`stepsFromEvents([
    { type: 'tool_start', tool: 'read_file', callId: 'r1', arguments: { file_path: '/skills/a/SKILL.md' } },
    { type: 'tool_result', tool: 'read_file', callId: 'r1' },
    { type: 'tool_start', tool: 'ls', callId: 'l1', arguments: { path: '/skills' } },
    { type: 'tool_result', tool: 'ls', callId: 'l1' },
    { type: 'tool_start', tool: 'glob', callId: 'g1', arguments: { pattern: '**/SKILL.md' } },
    { type: 'tool_result', tool: 'glob', callId: 'g1' },
    { type: 'tool_start', tool: 'grep', callId: 'p1', arguments: { pattern: 'query' } },
    { type: 'tool_result', tool: 'grep', callId: 'p1' },
    { type: 'tool_start', tool: 'get_track', callId: 'b1', arguments: { track_id: 7 } },
    { type: 'tool_result', tool: 'get_track', callId: 'b1' }
  ])`);
  assert.deepEqual(Array.from(steps, (step) => step.tool), ['get_track']);
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

test('confirmation remains actionable from the trace view and marks its tool as waiting', () => {
  const ui = page();
  ui.run(`currentSessionId = 't'; qaView = 'trace';
    const run = { id: 't', status: 'running', events: [
      { type: 'tool_start', tool: 'add_registry_vessel', label: '先验库写入', callId: 'c1', agent: 'executor' }
    ] };
    sessionRuns.set('t', run);
    dispatchRunEvent(run, { type: 'confirm', confirmation: ${JSON.stringify(confirmation)} });`);

  assert.equal(ui.elements.get('agentConfirmPanel').hidden, false);
  assert.equal(ui.run("sessionRuns.get('t').status"), 'awaiting_confirmation');
  assert.equal(ui.run("stepsFromEvents(sessionRuns.get('t').events)[0].status"), 'awaiting_confirmation');
  assert.equal(ui.elements.get('btnAskAgent').disabled, true);
  assert.equal(ui.elements.get('btnStopAgent').hidden, true);
});

test('confirmation card is mounted in the visible chat or trace view', () => {
  const confirmIndex = template.indexOf('id="agentConfirmPanel"');
  const chatMountIndex = template.indexOf('id="qaChatConfirmationMount"');
  const chatIndex = template.indexOf('id="qaChatView"');
  const traceMountIndex = template.indexOf('id="qaTraceConfirmationMount"');
  const traceIndex = template.indexOf('id="qaTraceView"');
  const chatAnswerIndex = template.indexOf('id="agentFinalView"');
  const traceListIndex = template.indexOf('id="qaTraceList"');

  assert.ok(chatIndex < chatAnswerIndex && chatAnswerIndex < chatMountIndex && chatMountIndex < confirmIndex);
  assert.ok(traceIndex < traceListIndex && traceListIndex < traceMountIndex);
  assert.match(template, /qa\.css\?v=18/);
  assert.match(template, /agent\.js\?v=89/);
  const ui = page();
  ui.run(`currentSessionId = 't'; qaView = 'chat';
    renderAgentConfirmation({ confirmation: ${JSON.stringify(confirmation)} });`);
  assert.equal(ui.elements.get('agentConfirmPanel').parentElement, ui.elements.get('qaChatConfirmationMount'));
  ui.run("switchQaView('trace')");
  assert.equal(ui.elements.get('agentConfirmPanel').parentElement, ui.elements.get('qaTraceConfirmationMount'));
  ui.run("switchQaView('chat')");
  assert.equal(ui.elements.get('agentConfirmPanel').parentElement, ui.elements.get('qaChatConfirmationMount'));
});

test('a pending confirmation survives a full conversation rerender', () => {
  const ui = page();
  const event = { type: 'confirm', confirmation, result: { state: 'awaiting_confirmation', confirmation } };
  ui.run(`currentSessionId = 't';
    currentSession = { sessionId: 't', turns: [] };
    sessionRuns.set('t', { id: 't', question: 'add 003', status: 'awaiting_confirmation',
      events: [${JSON.stringify(event)}], result: ${JSON.stringify(event.result)} });
    renderView();`);

  assert.equal(ui.elements.get('agentConfirmPanel').hidden, false);
  assert.equal(ui.elements.get('agentConfirmPanel').parentElement, ui.elements.get('qaChatConfirmationMount'));
  assert.equal(ui.elements.get('btnAskAgent').disabled, true);
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

test('history restores pending confirmation without saving runtime plans', () => {
  const ui = page();
  ui.run(`currentSessionId = 't'; renderRestoredTurn({
    state: 'awaiting_confirmation', confirmation: ${JSON.stringify(confirmation)}
  });`);
  assert.equal(ui.elements.get('agentPlanMeterFill').style.width, '0%');
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
  assert.equal(ui.elements.get('agentThinkingState').textContent, 'Closed with available evidence');
  assert.equal(ui.elements.get('agentPlanMeterFill').style.width, '0%');
  assert.match(ui.elements.get('agentAnswer').innerHTML, /kf1/);
  ui.run(`renderRestoredTurn({ state: 'completed', partial: true,
    answer: '已取得 kf1，尚未验证目标' });`);
  assert.equal(ui.elements.get('agentThinkingState').textContent, 'Closed with available evidence');
});

test('a stalled run is labeled incomplete instead of completed', () => {
  const ui = page();
  ui.run(`renderAgentAnswer({ state: 'stalled', answerText: 'Planner 未提交状态' });`);
  assert.match(ui.elements.get('agentAnswer').innerHTML, /Planner 未提交状态/);
  assert.equal(ui.elements.get('agentThinkingState').textContent, 'Incomplete');
  ui.run(`renderRestoredTurn({ state: 'stalled', answer: 'Planner 未提交状态' });`);
  assert.equal(ui.elements.get('agentThinkingState').textContent, 'Incomplete');
});

test('planner clarification is shown as a completed pause for user input', () => {
  const ui = page();
  ui.run(`renderAgentAnswer({ state: 'awaiting_input', success: true,
    answerText: '请补充时间范围。' });`);
  assert.match(ui.elements.get('agentAnswer').innerHTML, /请补充时间范围/);
  assert.equal(ui.elements.get('agentThinkingState').textContent, 'More information required');
  assert.equal(ui.elements.get('btnAskAgent').disabled, false);
});

test('an error shared by the event and result is displayed only once', () => {
  const ui = page();
  const detail = ui.run(`harnessErrorDetail({
    message: 'Planner update failed',
    result: { error: 'Planner update failed' }
  })`);
  assert.equal(detail, 'Planner update failed');
  assert.equal(ui.run(`harnessErrorDetail({
    message: 'Tool failed', error: 'timeout', result: { error: 'Tool failed' }
  })`), 'Tool failed\ntimeout');
});

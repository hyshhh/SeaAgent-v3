const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../web/static/js/agent.js'), 'utf8');
const template = fs.readFileSync(path.join(__dirname, '../../web/templates/index.html'), 'utf8');
const styles = fs.readFileSync(path.join(__dirname, '../../web/static/css/qa.css'), 'utf8');

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
    AbortController, URLSearchParams, console,
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

test('image presets fill the query and open the attachment picker', () => {
  const ui = page();
  const imageInput = {clickCount: 0, click() { this.clickCount += 1; }};
  const question = {value: '', focus() {}, style: {setProperty() {}}};
  ui.elements.set('agentImageInput', imageInput);
  ui.elements.set('agentQuestion', question);
  ui.run('useImageQuestion()');
  assert.equal(question.value, '视频中是否出现过与我上传图片相同的船？');
  assert.equal(imageInput.clickCount, 1);

  const result = ui.run(`(() => {
    const select = {
      selectedIndex: 3,
      selectedOptions: [{value: '根据我上传的图片查找视频中的船。', dataset: {attachImage: 'true'}}],
    };
    useAgentPreset(select);
    return {selectedIndex: select.selectedIndex, question: document.getElementById('agentQuestion').value};
  })()`);
  assert.equal(result.question, '根据我上传的图片查找视频中的船。');
  assert.equal(result.selectedIndex, 0);
  assert.equal(imageInput.clickCount, 2);
});

test('shared state renders escaped fields and its snapshot version', () => {
  const ui = page();
  ui.run(`renderSharedState({ version: 4, requirements: ['<script>'], decision: 'continue' });`);
  assert.match(ui.elements.get('qaSharedStateGroups').innerHTML, /&lt;script&gt;/);
  assert.match(ui.elements.get('qaSharedStateGroups').innerHTML, /Requirements/);
  assert.equal(ui.elements.get('qaSharedStateMeta').textContent, 'Version 4');
});

test('shared state groups render fields for every role and runtime section', () => {
  const ui = page();
  ui.run(`renderSharedState({
    evidence_ids: ['e1'], step_status: { step_1: 'completed' }, execution_status: 'success',
    assessment: { final_answer: 'done' }, evidence_gaps: ['none'], conflicts: [], decision: 'terminate',
    question: 'count ships', task_status: 'completed', iteration: 1, pending_agent: null,
    evidence: [{ id: 'e1' }], tool_results: [{ tool: 'dedup_tracks' }]
  });`);
  const html = ui.elements.get('qaSharedStateGroups').innerHTML;
  for (const group of ['executor', 'reflector', 'runtime', 'evidence']) assert.match(html, new RegExp(`data-group="${group}"`));
  for (const key of ['evidence_ids', 'assessment', 'task_status', 'tool_results']) assert.match(html, new RegExp(`data-key="${key}"`));
});

test('evidence statistics render even when the query returns no visual items', () => {
  const ui = page();
  ui.run(`renderEvidence({ statistics: {
    queryByTool: [{ tool: 'dedup_tracks', statistics: { trackCount: 0, highThresholdShipCount: 0, lowThresholdShipCount: 0 } }],
    display: { trackCount: 0, keyframeCount: 0, clipCount: 0, registryReferenceCount: 0 }
  } });`);
  const html = ui.elements.get('evidenceGallery').innerHTML;
  assert.match(html, /No track evidence available/);
  assert.match(html, /High-threshold entities: 0/);
  assert.doesNotMatch(html, /keyframes/i);
});

test('compact count result metadata remains visible in the evidence panel', () => {
  const ui = page();
  ui.run(`renderEvidence({ statistics: { queryByTool: [{
    tool: 'dedup_tracks',
    statistics: { trackCount: 20, highThresholdShipCount: 12, lowThresholdShipCount: 10, coverageComplete: false }
  }] } });`);
  const html = ui.elements.get('evidenceGallery').innerHTML;
  assert.match(html, /Raw tracks: 20/);
  assert.match(html, /High-threshold entities: 12/);
  assert.match(html, /Low-threshold entities: 10/);
  assert.match(html, /Search complete: No/);
});

test('registry count metadata distinguishes matched tracks from raw tracks', () => {
  const ui = page();
  ui.run(`renderEvidence({ statistics: { queryByTool: [{
    tool: 'match_entities',
    statistics: { trackCount: 24, inRegistryCount: 6, unregisteredCount: 18, coverageComplete: true }
  }] } });`);
  const html = ui.elements.get('evidenceGallery').innerHTML;
  assert.match(html, /Tracks matched: 24/);
  assert.match(html, /Tracks in registry: 6/);
  assert.match(html, /Tracks not in registry: 18/);
});

test('track evidence is one table with one keyframe per track and optional registry result', () => {
  const ui = page();
  ui.run(`renderEvidence({ registryMatchUsed: true, tools: ['match_entities'], trackRows: [
    { trackId: 't-1', startTimeText: '2026-10-25 10:00:00', endTimeText: '2026-10-25 10:00:05',
      startTime: 1792893600, endTime: 1792893605,
      keyframeId: 'kf-1',
      registryMatch: { status: 'In registry', score: 0.91, matchedObject: { hullNumber: '0857' } } },
    { trackId: 't-2', startTimeText: '2026-10-25 10:02:00', endTimeText: '2026-10-25 10:02:05',
      keyframeIds: ['kf-2-first', 'kf-2-second'],
      registryMatch: { status: 'Not in registry', score: 0.42, matchedObject: null } }
  ] });`);
  const html = ui.elements.get('evidenceGallery').innerHTML;
  assert.match(html, /<table class="qa-evidence-table">/);
  assert.match(html, /<th>Track ID<\/th><th>Appearance time<\/th><th>Keyframe<\/th><th>Registry match<\/th>/);
  assert.match(html, /src="\/api\/evidence\/keyframes\/kf-1\?scale=0.5"/);
  assert.match(html, /src="\/api\/evidence\/keyframes\/kf-2-first\?scale=0.5"/);
  assert.equal((html.match(/<img /g) || []).length, 2);
  assert.doesNotMatch(html, /kf-2-second|<video|autoplay|\/api\/evidence\/clips/);
  assert.match(html, /0857/);
  assert.match(html, /In registry/);
  assert.match(html, /Not in registry/);
  assert.doesNotMatch(html, /Registry references/);
});

test('non-match evidence leaves the registry column empty', () => {
  const ui = page();
  ui.run(`renderEvidence({ tools: ['dedup_tracks'], trackRows: [{
    trackId: 't-4', startTimeText: 'start', endTimeText: 'end'
  }] });`);
  const html = ui.elements.get('evidenceGallery').innerHTML;
  assert.match(html, /<td class="qa-evidence-registry-cell"><\/td>/);
  assert.doesNotMatch(html, /Not in registry|In registry/);
  ui.run(`renderEvidence({ registryMatchUsed: true, trackRows: [{trackId: 't-4'}] });`);
  assert.match(ui.elements.get('evidenceGallery').innerHTML, /<td class="qa-evidence-registry-cell"><\/td>/);
});

test('all shared-state fields use bounded scrollable value boxes', () => {
  assert.match(styles, /\.qa-state-field dd\s*\{/);
  assert.match(styles, /max-height: 120px/);
  assert.match(styles, /min-height: 36px/);
  assert.match(styles, /overflow: auto/);
  assert.match(styles, /border: 1px solid var\(--qa-line\)/);
  assert.doesNotMatch(styles, /qa-state-field\[data-key=/);
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

test('tool trace keeps compact placeholders and expandable arguments, returns and error type', () => {
  const ui = page();
  const steps = ui.run(`stepsFromEvents([
    { type: 'tool_start', tool: 'recognize_tracks', label: 'Direct recognition', callId: 'x1',
      arguments: { track_ids: ['t1', 't2'], description: 'yellow vessel' } },
    { type: 'tool_result', tool: 'recognize_tracks', label: 'Direct recognition', callId: 'x1', status: 'error',
      arguments: { track_ids: ['t1', 't2'], description: 'yellow vessel' },
      result: { ok: false, errorType: 'TimeoutError', error: 'tool_timeout' } }
  ])`);
  assert.equal(steps[0].status, 'error');
  assert.equal(steps[0].errorType, 'TimeoutError');
  assert.equal(JSON.stringify(steps[0].args.track_ids), '["t1","t2"]');
  assert.equal(steps[0].result.error, 'tool_timeout');
  ui.elements.set('qaTraceList', { innerHTML: '' });
  ui.elements.set('qaTraceRibbon', { innerHTML: '' });
  ui.elements.set('qaTraceSummary', { textContent: '' });
  ui.run(`currentSessionId = 'trace'; sessionRuns.set('trace', { id: 'trace', events: [
    { type: 'tool_start', tool: 'recognize_tracks', callId: 'x1', arguments: { track_ids: ['t1', 't2'], description: '${'vessel'.repeat(80)}' } },
    { type: 'tool_result', tool: 'recognize_tracks', callId: 'x1', status: 'error',
      result: { ok: false, errorType: 'TimeoutError', error: 'tool_timeout' } }
  ] }); renderTrajectory();`);
  const html = ui.elements.get('qaTraceList').innerHTML;
  const preview = html.match(/<div class="qa-step-args">([\s\S]*?)<\/div>/)?.[1] || '';
  assert.match(preview, /track_ids: 2 items/);
  assert.match(preview, /raw:.*…/);
  assert.ok(html.includes('vessel'.repeat(80)));
  assert.match(html, /Details/);
  assert.match(html, /TimeoutError/);
  assert.match(html, /qa-dot-error/);
  assert.match(styles, /#tab-agent-qa \.qa-step\.is-error \.qa-step-kind i/);
  assert.match(styles, /background: #d92d3a/);
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
  assert.match(template, /qa\.css\?v=21/);
  assert.match(template, /agent\.js\?v=92/);
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

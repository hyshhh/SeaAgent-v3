const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../web/static/js/settings.js'), 'utf8');

function page(apiFetch) {
  const elements = new Map([
    ['qaEndpointDialog', {open: false, showModal() { this.open = true; }, close() { this.open = false; }}],
    ['qaEndpointStatus', {textContent: ''}],
    ['qaEndpointSaveButton', {disabled: false}],
    ['qaVlmApiPort', {value: '', type: 'number'}],
    ['qaEmbeddingApiPort', {value: '', type: 'number'}],
  ]);
  const context = vm.createContext({
    document: {
      addEventListener() {},
      getElementById: (id) => elements.get(id) || null,
      querySelector: () => null,
      querySelectorAll: () => [],
    },
    window: {},
    apiFetch,
    showToast() {},
  });
  vm.runInContext(source, context);
  return {elements, run: (code) => vm.runInContext(code, context)};
}

const settings = {
  settings: {llm: {api_port: 7890}, embedding: {api_port: 7891}},
  specs: {
    'llm.api_port': {type: 'int', min: 1, max: 65535, step: 1},
    'embedding.api_port': {type: 'int', min: 1, max: 65535, step: 1},
  },
};

test('QA endpoint dialog reads and saves both model service ports', async () => {
  const requests = [];
  const ui = page(async (_url, options = {}) => {
    requests.push(options);
    return options.method === 'PUT' ? {data: settings} : settings;
  });

  await ui.run('openQaEndpointSettings()');
  assert.equal(ui.elements.get('qaEndpointDialog').open, true);
  assert.equal(ui.elements.get('qaVlmApiPort').value, 7890);
  assert.equal(ui.elements.get('qaEmbeddingApiPort').value, 7891);

  ui.elements.get('qaVlmApiPort').value = '9000';
  ui.elements.get('qaEmbeddingApiPort').value = '9001';
  await ui.run('saveQaEndpointSettings()');

  assert.equal(requests.length, 2);
  assert.deepEqual(JSON.parse(requests[1].body), {
    settings: {llm: {api_port: 9000}, embedding: {api_port: 9001}},
  });
});

test('QA endpoint dialog rejects out-of-range ports before saving', async () => {
  let requests = 0;
  const ui = page(async () => { requests += 1; return settings; });
  ui.elements.get('qaVlmApiPort').value = '70000';
  ui.elements.get('qaEmbeddingApiPort').value = '7891';

  await ui.run('saveQaEndpointSettings()');

  assert.equal(requests, 0);
  assert.match(ui.elements.get('qaEndpointStatus').textContent, /1 到 65535/);
});

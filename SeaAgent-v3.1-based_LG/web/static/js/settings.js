const SETTINGS_API = '/api/settings';

let systemSettings = {};
let systemSettingSpecs = {};

function settingValue(values, path) {
  return path.split('.').reduce((current, key) => current?.[key], values);
}

function setSettingValue(values, path, value) {
  const keys = path.split('.');
  let current = values;
  keys.slice(0, -1).forEach((key) => {
    if (!current[key] || typeof current[key] !== 'object') current[key] = {};
    current = current[key];
  });
  current[keys[keys.length - 1]] = value;
}

function settingsStatus(message, type = '') {
  const status = document.getElementById('settingsStatus');
  if (!status) return;
  status.textContent = message;
  status.className = `settings-status${type ? ` ${type}` : ''}`;
}

function updateSettingInput(input, value, spec = {}) {
  if (spec.type === 'bool' || input.type === 'checkbox') {
    input.checked = Boolean(value);
    return;
  }
  if (spec.type === 'text' || spec.type === 'string' || input.tagName === 'TEXTAREA') {
    input.value = value ?? '';
    if (spec.max_chars) input.maxLength = spec.max_chars;
    return;
  }
  if (spec.type === 'enum' || input.tagName === 'SELECT') {
    const next = value == null ? '' : String(value);
    if (Array.isArray(spec.choices) && spec.choices.length) {
      input.innerHTML = spec.choices.map(choice => {
        const optionValue = String(choice);
        return `<option value="${optionValue.replaceAll('&', '&amp;').replaceAll('"', '&quot;')}">${optionValue}</option>`;
      }).join('');
    }
    if (spec.choices && !spec.choices.map(String).includes(next) && next) {
      const option = document.createElement('option');
      option.value = next;
      option.textContent = next;
      input.appendChild(option);
    }
    input.value = next;
    return;
  }
  if (spec.min !== undefined) input.min = spec.min;
  if (spec.max !== undefined) input.max = spec.max;
  if (spec.step !== undefined) input.step = spec.step;
  input.value = value ?? '';
}

function fillSystemSettings(data) {
  systemSettings = data.settings || {};
  systemSettingSpecs = data.specs || {};
  document.querySelectorAll('[data-setting]').forEach((input) => {
    const path = input.dataset.setting;
    updateSettingInput(input, settingValue(systemSettings, path), systemSettingSpecs[path]);
  });
  syncAppearanceTrackingFields();
  syncMonitoringOptions(systemSettings);
}

function syncAppearanceTrackingFields() {
  const toggle = document.querySelector('[data-appearance-toggle]');
  const enabled = Boolean(toggle?.checked);
  document.querySelectorAll('[data-appearance-dependent] input').forEach((input) => {
    input.disabled = !enabled;
  });
  document.querySelectorAll('[data-appearance-dependent]').forEach((field) => {
    field.classList.toggle('is-disabled', !enabled);
  });
}

function setMonitoringSettingsStatus(message, state = 'synced') {
  const status = document.getElementById('monitoringSettingsStatus');
  if (!status) return;
  status.className = `monitor-settings-status ${state}`;
  status.innerHTML = `<i></i>${message}`;
}

function syncMonitoringOptions(settings) {
  const links = [
    [['optConf', 'camConf'], 'yolo.confidence'],
    [['optIou', 'camIou'], 'yolo.iou'],
    [['optDetectEvery', 'camDetectEvery'], 'yolo.detect_every_n_frames'],
    [['optTrackHigh'], 'yolo.tracker_params.track_high_thresh'],
    [['optTrackLow'], 'yolo.tracker_params.track_low_thresh'],
    [['optNewTrack'], 'yolo.tracker_params.new_track_thresh'],
    [['optMatchThresh'], 'yolo.tracker_params.match_thresh'],
    [['optTrackBuffer'], 'yolo.tracker_params.track_buffer'],
    [['optMaxStale'], 'pipeline.max_stale_frames'],
    [['optTargetFps', 'camTargetFps'], 'pipeline.target_fps'],
    [['optPipeScale', 'camPipeScale'], 'pipeline.pipe_scale'],
    [['optMaxFrames', 'camMaxFrames'], 'pipeline.max_frames'],
    [['optDevice', 'camDevice'], 'yolo.device'],
    [['optYoloModel', 'camYoloModel'], 'yolo.model'],
    [['optSaveVideo', 'camOptSaveVideo'], 'pipeline.save_output_video'],
  ];
  links.forEach(([elementIds, path]) => {
    const value = settingValue(settings, path);
    const spec = systemSettingSpecs[path] || {};
    elementIds.forEach((elementId) => {
      const input = document.getElementById(elementId);
      if (!input || value === undefined || value === null) return;
      updateSettingInput(input, value, spec);
    });
  });
  setMonitoringSettingsStatus('Synced with system settings', 'synced');
  const applyButton = document.getElementById('applyMonitoringSettingsButton');
  if (applyButton) applyButton.disabled = false;
}

async function saveMonitoringSettings() {
  const button = document.getElementById('applyMonitoringSettingsButton');
  try {
    const params = typeof collectVideoParams === 'function' ? collectVideoParams() : {};
    const settings = {
      yolo: {
        confidence: params.conf_threshold,
        iou: params.iou_threshold,
        detect_every_n_frames: params.detect_every,
        tracker_params: {
          track_high_thresh: params.track_high_thresh,
          track_low_thresh: params.track_low_thresh,
          new_track_thresh: params.new_track_thresh,
          match_thresh: params.match_thresh,
          track_buffer: params.track_buffer,
        },
        device: params.device,
        model: params.yolo_model,
      },
      pipeline: {
        target_fps: params.target_fps,
        pipe_scale: params.pipe_scale,
        max_frames: params.max_frames,
        max_stale_frames: params.max_stale_frames,
        save_output_video: params.save_output_video,
      },
    };
    if (button) {
      button.disabled = true;
      button.textContent = 'Applying...';
    }
    setMonitoringSettingsStatus('Saving changes...', 'saving');
    const response = await apiFetch(SETTINGS_API, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({settings}),
    });
    fillSystemSettings(response.data || response);
    setMonitoringSettingsStatus('Settings synchronized', 'synced');
    showToast('监控参数已同步到系统设置和后端，新任务将使用最新配置');
  } catch (error) {
    setMonitoringSettingsStatus('Synchronization failed', 'error');
    showToast(error.message, 'error');
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = 'Apply Settings';
    }
  }
}

function collectSystemSettings() {
  const values = {};
  for (const input of document.querySelectorAll('[data-setting]')) {
    const path = input.dataset.setting;
    const spec = systemSettingSpecs[path] || {};
    const raw = input.value;
    if (spec.type === 'bool' || input.type === 'checkbox') {
      setSettingValue(values, path, input.checked);
      continue;
    }
    if (spec.type === 'text' || spec.type === 'string' || input.tagName === 'TEXTAREA') {
      const text = String(raw || '').trim();
      if (!text && !spec.allow_empty) throw new Error(`参数“${spec.label || path}”不能为空`);
      if (spec.max_chars && text.length > spec.max_chars) {
        throw new Error(`提示词“${spec.label || path}”过长，最多 ${spec.max_chars} 字符`);
      }
      setSettingValue(values, path, text);
      continue;
    }
    if (spec.type === 'enum' || input.tagName === 'SELECT') {
      const choice = String(raw || '').trim();
      if (!choice) throw new Error(`参数「${path}」不能为空`);
      setSettingValue(values, path, choice);
      continue;
    }
    if (String(raw).trim() === '') throw new Error(`参数“${path}”不能为空`);
    const value = spec.type === 'int' ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
    if (!Number.isFinite(value)) throw new Error(`参数“${path}”必须是数字`);
    setSettingValue(values, path, value);
  }
  return values;
}

async function loadSystemSettings(showMessage = false) {
  settingsStatus('正在读取…');
  try {
    const data = await apiFetch(SETTINGS_API);
    fillSystemSettings(data);
    settingsStatus('已读取当前配置', 'success');
    if (showMessage) showToast('设置已重新读取');
  } catch (error) {
    settingsStatus('读取失败', 'error');
    if (showMessage) showToast(error.message, 'error');
  }
}

async function saveSystemSettings() {
  const button = document.getElementById('saveSettingsButton');
  try {
    const settings = collectSystemSettings();
    if (button) {
      button.disabled = true;
      button.textContent = '保存中…';
    }
    settingsStatus('正在保存…');
    const response = await apiFetch(SETTINGS_API, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({settings}),
    });
    fillSystemSettings(response.data || response);
    settingsStatus('已保存，新任务与问答生效', 'success');
    showToast('设置已保存，下一轮问答将使用最新提示词与阈值');
  } catch (error) {
    settingsStatus('保存失败', 'error');
    showToast(error.message, 'error');
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = '保存设置';
    }
  }
}

async function resetSystemSettings() {
  if (!window.confirm('确定恢复全部运行参数和提示词的默认值吗？')) return;
  const button = document.getElementById('saveSettingsButton');
  try {
    if (button) button.disabled = true;
    settingsStatus('正在恢复默认…');
    const response = await apiFetch(`${SETTINGS_API}/reset`, {method: 'POST'});
    fillSystemSettings(response.data || response);
    settingsStatus('已恢复默认值', 'success');
    showToast('运行参数与提示词已恢复默认值');
  } catch (error) {
    settingsStatus('恢复失败', 'error');
    showToast(error.message, 'error');
  } finally {
    if (button) button.disabled = false;
  }
}

window.loadSystemSettings = loadSystemSettings;
window.saveSystemSettings = saveSystemSettings;
window.resetSystemSettings = resetSystemSettings;
window.saveMonitoringSettings = saveMonitoringSettings;

document.addEventListener('DOMContentLoaded', () => loadSystemSettings(false));


document.addEventListener('change', (event) => {
  if (event.target?.matches?.('[data-appearance-toggle]')) syncAppearanceTrackingFields();
  if (event.target?.closest?.('.monitoring-settings-panel')) {
    setMonitoringSettingsStatus('Unsaved changes', 'dirty');
  }
});

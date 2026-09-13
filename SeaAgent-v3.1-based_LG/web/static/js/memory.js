let memoryRefreshTimer = null;
let memorySettingsLoaded = false;

function formatMemoryTime(timestamp) {
  const value = Number(timestamp);
  if (!Number.isFinite(value)) return 'Unknown time';
  if (value < 946684800) return `Video ${formatVideoTime(value)}`;
  const date = new Date(value * 1000);
  const pad = number => String(number).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function escapeMemoryText(value) {
  return String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function memoryMatchKey(value) {
  const keys = {
    confirmed: 'confirmed',
    '已确认': 'confirmed',
    candidate: 'candidate',
    '候选': 'candidate',
    conflict: 'conflict',
    '冲突': 'conflict',
    unknown: 'unknown',
    '未知': 'unknown',
  };
  return keys[String(value ?? '').trim().toLowerCase()] || 'unknown';
}

function memoryMatchLabel(value) {
  const labels = {confirmed:'Confirmed', candidate:'Candidate', conflict:'Conflict', unknown:'Unknown'};
  return labels[memoryMatchKey(value)] || 'Unknown';
}

function memoryStateKey(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  return ['complete', 'completed', 'done', '已完成', '完成'].includes(normalized) ? 'done' : 'building';
}

function memoryStateLabel(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (['complete', 'completed', 'done', '已完成', '完成'].includes(normalized)) return 'Complete';
  if (['building', 'processing', 'active', '构建中', '处理中', '建立中'].includes(normalized)) return 'Building';
  return 'Unknown';
}

function renderTrackMemory(payload) {
  document.getElementById('memoryTrackCount').textContent = payload.trackCount ?? 0;
  document.getElementById('memoryKeyframeCount').textContent = payload.keyframeCount ?? 0;
  document.getElementById('memoryEmbeddedCount').textContent = payload.embeddedKeyframeCount ?? 0;
  if (!memorySettingsLoaded) {
    const retentionSeconds = Number(payload.settings?.retentionSeconds ?? 0);
    document.getElementById('memoryRetentionHours').value = Math.round(retentionSeconds / 3600 * 10) / 10;
    memorySettingsLoaded = true;
  }
  const body = document.getElementById('trackMemoryTable');
  const tracks = payload.tracks || [];
  if (!tracks.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-msg memory-empty">No track records yet. Process a video in Maritime Monitoring first.</td></tr>';
    return;
  }
  body.innerHTML = tracks.map(track => {
    const hull = track.finalHullNumber || 'No stable hull';
    const frames = `${track.embeddedKeyframeCount || 0}/${track.keyframeCount || 0}`;
    const matchKey = memoryMatchKey(track.finalMatchType);
    const stateClass = memoryStateKey(track.memoryState);
    const description = escapeMemoryText(track.finalDescription || 'No stable description');
    return `<tr>
      <td><strong class="memory-track-id">${escapeMemoryText(track.trackId)}</strong></td>
      <td class="memory-time-range">${formatMemoryTime(track.startTime)}<span>—</span>${formatMemoryTime(track.endTime)}</td>
      <td class="memory-hull">${escapeMemoryText(hull)}</td>
      <td><span class="memory-state ${matchKey}">${escapeMemoryText(memoryMatchLabel(track.finalMatchType))}</span></td>
      <td><span class="memory-frame-count">${frames}</span><small>embedded / total</small></td>
      <td><span class="memory-state ${stateClass}">${escapeMemoryText(memoryStateLabel(track.memoryState))}</span></td>
      <td class="memory-description" title="${description}"><div class="memory-description-text">${description}</div></td>
    </tr>`;
  }).join('');
}

async function loadTrackMemory(silent = false) {
  const state = document.getElementById('memoryRefreshState');
  try {
    if (!silent) state.textContent = 'Refreshing…';
    const payload = await apiFetch('/api/memory/tracks');
    renderTrackMemory(payload);
    state.textContent = `Updated ${new Date().toLocaleTimeString('zh-CN', {hour12:false})}`;
  } catch (error) {
    state.textContent = 'Refresh failed';
    if (!silent) showToast(error.message, 'error');
  }
}

function startMemoryAutoRefresh() {
  stopMemoryAutoRefresh();
  memoryRefreshTimer = setInterval(() => loadTrackMemory(true), 2000);
}

function stopMemoryAutoRefresh() {
  if (memoryRefreshTimer) clearInterval(memoryRefreshTimer);
  memoryRefreshTimer = null;
}

async function saveMemorySettings() {
  const input = document.getElementById('memoryRetentionHours');
  const hours = Number(input.value);
  if (!Number.isFinite(hours) || hours < 0 || hours > 24) {
    showToast('Enter a retention period from 0 to 24 hours.', 'error');
    return;
  }
  const retentionSeconds = Math.round(hours * 3600);
  try {
    const response = await apiFetch('/api/memory/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({retention_seconds: retentionSeconds})
    });
    memorySettingsLoaded = false;
    showToast('Memory retention updated.');
    await loadTrackMemory(true);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function clearAllMemory() {
  if (!confirm('Clear all trajectory memory? This cannot be undone. The registry will not be affected.')) return;
  try {
    const response = await apiFetch('/api/memory', {method: 'DELETE'});
    if (typeof clearTrajectoryEvidenceView === 'function') clearTrajectoryEvidenceView();
    showToast('Trajectory memory cleared.');
    await loadTrackMemory(true);
    if (typeof loadAgentMemorySummary === 'function') loadAgentMemorySummary();
  } catch (error) {
    showToast(error.message, 'error');
  }
}

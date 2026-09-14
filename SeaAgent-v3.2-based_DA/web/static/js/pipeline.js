/**
 * SeaAgent 监控前端逻辑 — 视频与摄像头推流
 *
 * 视频 Demo：后端推理，实时 MJPEG 推流到前端，不保存输出视频
 * 摄像头 Demo：浏览器/服务器摄像头，实时推流识别
 */

const PIPE_API = '/api/pipeline';

document.addEventListener('DOMContentLoaded', () => {
  const videoTab = document.getElementById('tab-monitoring');
  if (videoTab?.classList.contains('active')) {
    initializeContinuousMonitorControls();
    loadVideoList();
    loadTaskHistory();
  }
});

// ── Tab 切换 ──
function switchTab(tabName) {
  if (typeof stopMemoryAutoRefresh === 'function') stopMemoryAutoRefresh();
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-content').forEach(el => {
    el.classList.toggle('active', el.id === `tab-${tabName}`);
  });
  // 按需加载数据
  if (tabName === 'monitoring') {
    loadVideoList();
    loadTaskHistory();
  } else if (tabName === 'memory') {
    if (typeof loadTrackMemory === 'function') loadTrackMemory();
    if (typeof startMemoryAutoRefresh === 'function') startMemoryAutoRefresh();
  } else if (tabName === 'agent-qa') {
    if (typeof loadAgentMemorySummary === 'function') loadAgentMemorySummary();
  } else if (tabName === 'camera-demo') {
    onCameraSourceChange();
  } else if (tabName === 'database') {
    if (typeof loadShips === 'function') loadShips();
  } else if (tabName === 'settings') {
    if (typeof loadSystemSettings === 'function') loadSystemSettings();
  }
}

// ═══════════════════════════════════════════
// 视频 Demo
// ═══════════════════════════════════════════

let selectedVideo = null;
let currentTaskId = null;
let statusPollTimer = null;
let poolPollTimer = null;
let streamWs = null;        // WebSocket 推流连接
let _h264Ws = null;          // H.264 WebSocket
let _h264MediaSource = null; // MediaSource
let _h264SourceBuffer = null;// SourceBuffer
let _h264ObjectUrl = null;   // MediaSource 对象地址
let _h264Queue = [];         // 积压的 segment 队列
let videoListLoading = false;
let videoCatalog = [];
let videoDirectories = [];
let selectedVideoDirectory = '';
let continuousSelectedVideos = new Set();
let continuousMonitorState = null;

// ── 视频目录与列表 ──
function getVideoDirectory(video) {
  if (typeof video.directory === 'string') return video.directory;
  const filename = String(video.filename || '');
  const slashIndex = filename.indexOf('/');
  return slashIndex >= 0 ? filename.slice(0, slashIndex) : '';
}

function buildVideoDirectories(videos) {
  const counts = new Map([['', 0]]);
  videos.forEach(video => {
    const directory = getVideoDirectory(video);
    counts.set(directory, (counts.get(directory) || 0) + 1);
  });
  return [
    { path: '', name: 'Root Directory', count: counts.get('') || 0 },
    ...[...counts.keys()]
      .filter(path => path)
      .sort((a, b) => a.localeCompare(b))
      .map(path => ({ path, name: path, count: counts.get(path) || 0 })),
  ];
}

function renderVideoDirectorySelector() {
  const select = document.getElementById('videoDirectorySelect');
  const summary = document.getElementById('videoDirectorySummary');
  if (!select) return;

  select.innerHTML = videoDirectories.map(directory => `
    <option value="${safeAttr(directory.path)}">${escHtml(directory.name)} (${directory.count})</option>
  `).join('');
  select.value = selectedVideoDirectory;

  const current = videoDirectories.find(directory => directory.path === selectedVideoDirectory);
  if (summary) {
    summary.textContent = current
      ? `${current.count} Videos Available`
      : 'No videos in this directory';
  }
}

function getVisibleVideoCatalog() {
  return videoCatalog.filter(video => getVideoDirectory(video) === selectedVideoDirectory);
}

function renderVideoList() {
  const container = document.getElementById('videoList');
  if (!container) return;
  const visibleVideos = getVisibleVideoCatalog();

  if (!visibleVideos.length) {
    const message = selectedVideoDirectory
      ? '该子目录下暂无视频'
      : (videoCatalog.length ? '根目录下暂无视频，请选择一个子目录' : '暂无视频');
    container.innerHTML = `<div class="empty-msg">${message}</div>`;
    updateContinuousMonitorControls();
    return;
  }

  container.innerHTML = visibleVideos.map((v, index) => {
    const directory = getVideoDirectory(v);
    const relativeName = directory && v.filename.startsWith(`${directory}/`)
      ? v.filename.slice(directory.length + 1)
      : v.filename;
    return `
      <div class="video-item ${selectedVideo === v.filename ? 'selected' : ''} ${continuousSelectedVideos.has(v.filename) ? 'sequence-selected' : ''}"
           onclick="selectVideo(this.dataset.name, this)" data-name="${safeAttr(v.filename)}">
        <label class="video-sequence-check" onclick="event.stopPropagation()" title="加入连续监控序列"><input type="checkbox" data-name="${safeAttr(v.filename)}" ${continuousSelectedVideos.has(v.filename) ? 'checked' : ''} onchange="toggleContinuousVideo(this.dataset.name, this.checked)"></label>
        <span class="video-sequence-index">${index + 1}</span>
        <div class="video-item-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M8 5v14M16 5v14M3 10h5M16 10h5M3 14h5M16 14h5"/></svg></div>
        <div class="video-item-info">
           <div class="video-item-name">${escHtml(relativeName)}</div>
           <div class="video-item-meta">${v.size_mb == null ? 'Video File' : `${v.size_mb} MB`}<span class="video-sequence-state" data-sequence-state="${safeAttr(v.filename)}">${continuousSelectedVideos.has(v.filename) ? 'Queued' : 'Not Selected'}</span></div>
        </div>
        <div class="video-item-actions">
          <button class="video-delete-button" title="Delete video" aria-label="Delete video" onclick="event.stopPropagation(); deleteVideo(this.dataset.name)" data-name="${safeAttr(v.filename)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5"/></svg></button>
        </div>
      </div>
    `;
  }).join('');
  updateContinuousMonitorControls();
}

function changeVideoDirectory(directory) {
  if (directory === selectedVideoDirectory) return;
  if (continuousMonitorState) {
    showToast('连续监控进行中，请先停止当前序列', 'info');
    renderVideoDirectorySelector();
    return;
  }
  if (currentTaskId) {
    if (!confirm('当前有 Pipeline 正在运行，切换目录将停止当前任务。是否继续？')) {
      renderVideoDirectorySelector();
      return;
    }
    stopVideoPipeline();
  }

  // 目录是连续监控序列的边界，切换目录时清空上一目录的选择，避免跨目录混用。
  continuousSelectedVideos.clear();
  selectedVideo = null;
  selectedVideoDirectory = directory;
  // 切换目录只重置当前视频和结果状态，不隐藏推流主面板。
  const pipelinePanel = document.getElementById('pipelineControl');
  if (pipelinePanel) pipelinePanel.style.display = '';
  _restoreResultPlaceholder();
  resetPipelineStatus();
  renderVideoDirectorySelector();
  renderVideoList();
}

async function loadVideoList() {
  const container = document.getElementById('videoList');
  if (!container) return;
  if (videoListLoading) return;
  videoListLoading = true;
  container.innerHTML = '<div class="empty-msg">正在读取视频目录…</div>';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const resp = await fetch(`${PIPE_API}/videos`, { signal: controller.signal });
    if (!resp.ok) throw new Error(`请求失败 (${resp.status})`);
    const data = await resp.json();
    videoCatalog = data.videos || [];
    videoDirectories = data.directories || buildVideoDirectories(videoCatalog);
    if (!videoDirectories.some(directory => directory.path === selectedVideoDirectory)) {
      selectedVideoDirectory = '';
    }
    const availableNames = new Set(videoCatalog.map(video => video.filename));
    continuousSelectedVideos = new Set([...continuousSelectedVideos].filter(name => availableNames.has(name)));
    renderVideoDirectorySelector();
    renderVideoList();
  } catch (e) {
    const message = e.name === 'AbortError' ? '目录响应超时，请检查视频盘是否已挂载' : e.message;
    container.innerHTML = `<div class="empty-msg">加载失败：${escHtml(message)}<br><button class="btn btn-sm" onclick="loadVideoList()">重新加载</button></div>`;
  } finally {
    clearTimeout(timeout);
    videoListLoading = false;
  }
}

function selectVideo(filename, el) {
  if (continuousMonitorState) {
    showToast('连续监控进行中，请先停止当前序列', 'info');
    return;
  }
  // 如果有 pipeline 在运行，先提示用户
  if (currentTaskId) {
    if (!confirm('当前有 Pipeline 正在运行，切换视频将停止当前任务。是否继续？')) return;
    stopVideoPipeline();
  }

  selectedVideo = filename;
  document.getElementById('pipelineControl').style.display = '';
  // 更新选中状态
  document.querySelectorAll('.video-item').forEach(item => item.classList.remove('selected'));
  if (el) el.classList.add('selected');

  showVideoPreview(filename);
  resetPipelineStatus();
}

/** 显示所选视频的预览帧 */
function showVideoPreview(filename) {
  const resultPlaceholder = document.getElementById('resultPlaceholder');
  if (!resultPlaceholder || !filename) return;

  resultPlaceholder.innerHTML = '';
  resultPlaceholder.className = 'video-preview';
  resultPlaceholder.style.cssText = '';
  resultPlaceholder.style.display = '';

  const image = document.createElement('img');
  image.className = 'video-preview-image';
  image.alt = `${filename} 视频预览`;
  image.src = `${PIPE_API}/video-preview/${encodeURIComponent(filename)}`;
  image.onerror = () => {
    if (selectedVideo !== filename) return;
    resultPlaceholder.innerHTML = '<span>⚠️</span><p>视频预览加载失败，仍可开始处理</p>';
    resultPlaceholder.className = 'video-placeholder';
  };

  const label = document.createElement('div');
  label.className = 'video-preview-label';
  label.textContent = `视频预览 · ${filename}`;
  resultPlaceholder.append(image, label);
}

async function deleteVideo(filename) {
  if (continuousMonitorState) {
    showToast('连续监控进行中，暂不能删除视频', 'info');
    return;
  }
  if (!confirm(`确定删除视频 "${filename}"？`)) return;
  try {
    const resp = await fetch(`${PIPE_API}/videos/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '删除失败');
    showToast('已删除: ' + filename);
    continuousSelectedVideos.delete(filename);
    if (selectedVideo === filename) {
      selectedVideo = null;
      document.getElementById('pipelineControl').style.display = 'none';
      _restoreResultPlaceholder();
    }
    loadVideoList();
  } catch (e) {
    showToast(e.message, 'error');
  }
}

// ── Pipeline 控制 ──

/** 收集视频 Demo 页的 pipeline 参数 */
function collectVideoParams() {
  return {
    conf_threshold: parseFloat(document.getElementById('optConf').value) || 0.5,
    iou_threshold: parseFloat(document.getElementById('optIou').value) || 0.5,
    detect_every: parseInt(document.getElementById('optDetectEvery').value, 10) || 1,
    track_high_thresh: parseFloat(document.getElementById('optTrackHigh').value),
    track_low_thresh: parseFloat(document.getElementById('optTrackLow').value),
    new_track_thresh: parseFloat(document.getElementById('optNewTrack').value),
    match_thresh: parseFloat(document.getElementById('optMatchThresh').value),
    track_buffer: parseInt(document.getElementById('optTrackBuffer').value, 10),
    max_stale_frames: parseInt(document.getElementById('optMaxStale').value, 10),
    target_fps: parseFloat(document.getElementById('optTargetFps').value) || 0,
    pipe_scale: parseFloat(document.getElementById('optPipeScale').value) || 0.25,
    save_output_video: document.getElementById('optSaveVideo').checked,
    max_frames: parseInt(document.getElementById('optMaxFrames').value, 10) || 0,
    device: document.getElementById('optDevice').value.trim(),
    yolo_model: document.getElementById('optYoloModel').value.trim(),
  };
}

/** 收集摄像头页的 pipeline 参数 */
function collectCameraParams() {
  return {
    conf_threshold: parseFloat(document.getElementById('camConf').value) || 0.5,
    iou_threshold: parseFloat(document.getElementById('camIou').value) || 0.5,
    detect_every: parseInt(document.getElementById('camDetectEvery').value, 10) || 2,
    target_fps: parseFloat(document.getElementById('camTargetFps').value) || 0,
    capture_fps: parseInt(document.getElementById('camCaptureFps').value, 10) || 15,
    pipe_scale: parseFloat(document.getElementById('camPipeScale')?.value) || 0.25,
    save_output_video: document.getElementById('camOptSaveVideo').checked,
    max_frames: parseInt(document.getElementById('camMaxFrames').value, 10) || 0,
    device: document.getElementById('camDevice').value.trim(),
    yolo_model: document.getElementById('camYoloModel').value.trim(),
    stream_mode: (document.getElementById('camStreamMode') || {}).value || 'mjpeg',
  };
}

async function startVideoPipeline() {
  if (!selectedVideo) { showToast('请先选择视频', 'error'); return; }
  await launchVideoPipeline(selectedVideo, null, false);
}

async function launchVideoPipeline(filename, monitorStartTime = null, sequenceMode = false, sequenceOptions = null) {

  const btn = document.getElementById('btnStartPipeline');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Starting...';
  }

  try {
    const resp = await fetch(`${PIPE_API}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_filename: filename,
        video_filenames: sequenceOptions?.filenames || null,
        monitor_start_time: monitorStartTime,
        segment_gap_seconds: sequenceOptions?.gapSeconds || 0,
        playlist_failure_policy: sequenceOptions?.failurePolicy || 'skip',
        ...collectVideoParams(),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '启动失败');

    currentTaskId = data.task_id;
    if (continuousMonitorState && sequenceMode) continuousMonitorState.currentTaskId = currentTaskId;
    if (!sequenceMode) showToast(`流水线已启动 (${currentTaskId})`);
    updatePipelineStatus('running', sequenceMode ? `Processing sequence: ${filename}` : 'Processing...');
    const startButton = document.getElementById('btnStartPipeline');
    const stopButton = document.getElementById('btnStopPipeline');
    if (startButton) startButton.style.display = 'none';
    if (stopButton) stopButton.style.display = '';

    // 实时预览：H.264 WebSocket 推流 + MSE 播放
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    if (resultPlaceholder) {
      resultPlaceholder.className = 'stream-preview';
      resultPlaceholder.innerHTML = `
        <video id="streamVideo" class="demo-video" autoplay muted playsinline></video>
        <div id="streamFps" class="stream-status">正在连接推流...</div>
      `;
      resultPlaceholder.style.cssText = '';
    }

    connectStreamWs(currentTaskId);
    startStatusPolling();
    return true;
  } catch (e) {
    if (!sequenceMode) showToast('启动失败: ' + e.message, 'error');
    else showToast(`启动失败：${filename}`, 'error');
    return false;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '▶ Start Processing';
    }
    updateContinuousMonitorControls();
  }
}

async function startContinuousMonitoring() {
  if (continuousMonitorState) return;
  const queue = getVisibleVideoCatalog()
    .map(video => video.filename)
    .filter(filename => continuousSelectedVideos.has(filename));
  if (!queue.length) {
    showToast('请先勾选要连续监控的视频，或点击全选', 'error');
    return;
  }
  const startInput = document.getElementById('continuousStartTime');
  const parsedStart = startInput?.value ? new Date(startInput.value).getTime() / 1000 : null;
  if (startInput?.value && (!Number.isFinite(parsedStart) || parsedStart <= 0)) {
    showToast('监控起始时间格式不正确', 'error');
    return;
  }
  const parsedGap = Number(document.getElementById('continuousGapSeconds')?.value || 0);
  const gapSeconds = Number.isFinite(parsedGap) ? Math.max(0, parsedGap) : 0;
  const failurePolicy = document.getElementById('continuousFailurePolicy')?.value || 'skip';
  continuousMonitorState = {
    queue,
    index: 0,
    gapSeconds,
    failurePolicy,
    currentTaskId: null,
    cancelled: false,
    results: queue.map((filename, index) => ({index, filename, status: 'queued'})),
  };
  queue.forEach(filename => markSequenceVideo(filename, 'queued'));
  selectedVideo = queue[0];
  markSequenceVideo(queue[0], 'running');
  clearPoolTables();
  updateContinuousProgress(`Initializing one pipeline · 0 / ${queue.length} videos`);
  updateContinuousMonitorControls();
  const started = await launchVideoPipeline(queue[0], parsedStart, true, {
    filenames: queue,
    gapSeconds,
    failurePolicy,
  });
  if (!started) finishContinuousMonitoring('Sequence failed to start', 'failed');
}

function syncContinuousSequenceStatus(data) {
  const state = continuousMonitorState;
  if (!state || state.currentTaskId !== currentTaskId) return;
  const results = Array.isArray(data.playlist_results) ? data.playlist_results : [];
  results.forEach((result) => {
    if (!result?.filename || !result.status || result.status === 'queued') return;
    markSequenceVideo(result.filename, result.status);
  });
  state.results = results.length ? results : state.results;
  const index = Number(data.playlist_index);
  if (Number.isInteger(index) && index >= 0) state.index = index;
  const current = data.playlist_current;
  if (current && data.playlist_segment_status === 'running') {
    selectedVideo = current;
    markSequenceVideo(current, 'running');
  }
  if (data.progress) updateContinuousProgress(data.progress);
}

function finishContinuousMonitoring(message, status = 'completed') {
  continuousMonitorState = null;
  updateContinuousMonitorControls();
  resetPipelineButtons();
  updatePipelineStatus(status, message);
  updateContinuousProgress(message);
  showToast(message, status === 'completed' ? 'success' : 'info');
  loadTaskHistory();
}

async function stopContinuousMonitoring() {
  if (!continuousMonitorState) return;
  continuousMonitorState.cancelled = true;
  if (currentTaskId) {
    await stopVideoPipeline();
  } else {
    finishContinuousMonitoring('连续监控已停止', 'failed');
  }
}

/** 建立 H.264 WebSocket 推流连接（MSE 播放） */
function connectStreamWs(taskId) {
  disconnectStreamWs();

  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${wsProto}://${location.host}${PIPE_API}/ws/h264/${taskId}`;

  const videoEl = document.getElementById('streamVideo');
  if (!videoEl) return;

  // MediaSource
  const ms = new MediaSource();
  _h264ObjectUrl = URL.createObjectURL(ms);
  videoEl.src = _h264ObjectUrl;
  videoEl.load();  // 强制加载，确保 sourceopen 触发
  _h264MediaSource = ms;
  _h264SourceBuffer = null;
  _h264Queue = [];

  ms.addEventListener('sourceopen', () => {
    if (_h264MediaSource !== ms) return;
    // 等 WebSocket 收到 init segment 后再添加 SourceBuffer
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    streamWs = ws;
    _h264Ws = ws;

    let segmentCount = 0;
    let segmentTimer = performance.now();
    let segmentRate = null;

    function _updateStreamStatus() {
      const statusEl = document.getElementById('streamFps');
      if (!statusEl) return;
      const resolution = videoEl.videoWidth > 0 ? `${videoEl.videoWidth}×${videoEl.videoHeight}` : '读取中';
      const rate = segmentRate === null ? '' : ` · 更新 ${segmentRate} 次/秒`;
      statusEl.textContent = `推流分辨率 ${resolution}${rate}`;
    }

    function _syncToLiveEdge() {
      if (!videoEl.buffered.length) return;
      const liveEdge = videoEl.buffered.end(videoEl.buffered.length - 1);
      if (liveEdge - videoEl.currentTime > 1.0) {
        videoEl.currentTime = Math.max(0, liveEdge - 0.15);
      }
    }

    videoEl.addEventListener('loadedmetadata', _updateStreamStatus);
    videoEl.addEventListener('resize', _updateStreamStatus);

    /** 尝试播放（autoplay 可能被浏览器策略阻止） */
    function _ensurePlay() {
      const vEl = document.getElementById('streamVideo');
      if (vEl && vEl.paused) {
        vEl.play().catch(() => {
          // autoplay 被阻止，用户点击视频区域可手动播放
          vEl.muted = true;
          vEl.play().catch(() => {});
        });
      }
    }

    /** 处理队列积压 + 清理已播放缓冲区 */
    function _processQueue() {
      const sb = _h264SourceBuffer;
      if (!sb || sb.updating) return;

      _syncToLiveEdge();

      // 清理已播放的旧缓冲区（保留播放位置前 3 秒）
      try {
        const vEl = document.getElementById('streamVideo');
        if (vEl && sb.buffered.length > 0 && sb.buffered.start(0) < vEl.currentTime - 8) {
          sb.remove(sb.buffered.start(0), vEl.currentTime - 3);
          return; // remove 完成后会再次触发 updateend
        }
      } catch (e) {}

      // 追加队列中下一个分片
      if (_h264Queue.length > 0) {
        try {
          sb.appendBuffer(_h264Queue.shift());
        } catch (e) {
          console.warn('主推流缓冲异常，重新建立解码连接:', e);
          if (ws.readyState === WebSocket.OPEN) ws.close(1013, '解码缓冲积压');
        }
      }
    }

    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        const view = new DataView(evt.data);
        const msgType = view.getUint8(0);
        const payload = evt.data.slice(5);

        if (msgType === 0x01) {
          // Init segment (moov) — 创建 SourceBuffer（仅首次）
          if (_h264SourceBuffer) {
            // 同一连接重复收到初始化段说明编码上下文已变化，完整重连
            if (ws.readyState === WebSocket.OPEN) ws.close(1012, '编码上下文已更新');
            return;
          }
          try {
            if (ms.readyState !== 'open') {
              console.warn('MediaSource 未就绪，忽略 init segment');
              return;
            }
            const codecs = 'avc1.42C01F'; // H.264 Constrained Baseline Level 3.1
            const sb = ms.addSourceBuffer(`video/mp4; codecs="${codecs}"`);
            _h264SourceBuffer = sb;

            sb.addEventListener('updateend', () => {
              _processQueue();
            });
            sb.addEventListener('error', (e) => {
              console.error('SourceBuffer 错误:', e);
            });

            sb.appendBuffer(payload);
            _ensurePlay();  // init segment 就绪后尝试播放
          } catch (e) {
            console.error('MSE SourceBuffer 创建失败:', e);
          }

        } else if (msgType === 0x02) {
          // Media segment (moof+mdat)
          const sb = _h264SourceBuffer;
          if (!sb) return;

          if (sb.updating) {
            // 中间分片不可跳过；积压过多时完整重连并等待下一个关键帧
            if (_h264Queue.length >= 24) {
              ws.close(1013, '解码队列积压');
              return;
            }
            _h264Queue.push(payload);
          } else {
            try {
              sb.appendBuffer(payload);
              if (segmentCount === 0) _ensurePlay();  // 首个媒体段到达后尝试播放
            } catch (e) {
              console.warn('主推流分片追加失败，重新建立解码连接:', e);
              if (ws.readyState === WebSocket.OPEN) ws.close(1013, '分片追加失败');
            }
          }

          // 统计媒体分片到达速率，不再将其误标为视频帧率
          segmentCount++;
          const now = performance.now();
          if (now - segmentTimer > 1000) {
            segmentRate = (segmentCount * 1000 / (now - segmentTimer)).toFixed(1);
            _updateStreamStatus();
            segmentCount = 0;
            segmentTimer = now;
          }
        }
      } else {
        // JSON 控制消息
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'done') {
            disconnectStreamWs();
            const fpsEl = document.getElementById('streamFps');
            if (fpsEl) fpsEl.textContent = '处理完成';
          }
        } catch {}
      }
    };

    ws.onclose = () => {
      if (currentTaskId === taskId) {
        _scheduleReconnect('h264-stream', () => {
          if (currentTaskId === taskId) connectStreamWs(taskId);
        }, taskId);
      }
    };

    ws.onerror = () => {};
  });
}

/** 断开 H.264 推流 */
function disconnectStreamWs() {
  _clearReconnect('h264-stream');
  if (_h264Ws) {
    _h264Ws.onclose = null;
    _h264Ws.close();
    _h264Ws = null;
  }
  if (streamWs) {
    streamWs.onclose = null;
    streamWs.close();
    streamWs = null;
  }
  // 完整释放旧解码上下文，重连时重新创建 MediaSource
  if (_h264SourceBuffer) {
    try { if (_h264SourceBuffer.updating) _h264SourceBuffer.abort(); } catch {}
  }
  if (_h264MediaSource && _h264MediaSource.readyState === 'open') {
    try { _h264MediaSource.endOfStream(); } catch {}
  }
  _h264MediaSource = null;
  _h264SourceBuffer = null;
  _h264Queue = [];

  const videoEl = document.getElementById('streamVideo');
  if (videoEl) {
    videoEl.pause();
    videoEl.removeAttribute('src');
    videoEl.load();
  }
  if (_h264ObjectUrl) {
    URL.revokeObjectURL(_h264ObjectUrl);
    _h264ObjectUrl = null;
  }
}

async function stopVideoPipeline() {
  if (!currentTaskId) return;
  const taskId = currentTaskId;
  const stoppingSequence = Boolean(continuousMonitorState);

  // 立即停止轮询，防止后续 pollTaskStatus 干扰新任务
  stopStatusPolling();
  currentTaskId = null;
  clearPoolTables();

  // 断开 WebSocket 推流
  disconnectStreamWs();

  // 更新 UI 状态
  updatePipelineStatus('failed', 'Stopping...');
  resetPipelineButtons();

  try {
    const resp = await fetch(`${PIPE_API}/stop/${taskId}`, { method: 'POST' });
    if (resp.ok || resp.status === 404) {
      showToast('已停止');
    } else {
      const data = await resp.json().catch(() => ({}));
      showToast('停止: ' + (data.message || '完成'), 'info');
    }
  } catch (e) {
    showToast('已停止', 'info');
  }

  // 恢复结果占位
  _restoreResultPlaceholder();

  loadTaskHistory();
  if (stoppingSequence) finishContinuousMonitoring('连续监控已停止', 'failed');
}

function startStatusPolling() {
  stopStatusPolling();
  statusPollTimer = setInterval(pollTaskStatus, 2000);
  const logBox = document.getElementById('pipelineLogBox');
  if (logBox) logBox.style.display = '';
  clearPoolTables();
  pollPoolStatus();
  poolPollTimer = setInterval(pollPoolStatus, 1200);
}

function stopStatusPolling() {
  if (statusPollTimer) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }
  if (poolPollTimer) {
    clearInterval(poolPollTimer);
    poolPollTimer = null;
  }
}

async function pollTaskStatus() {
  // 快照当前任务 ID，防止请求返回时 currentTaskId 已变为新任务
  const taskId = currentTaskId;
  if (!taskId) return;
  try {
    const resp = await fetch(`${PIPE_API}/status/${taskId}`);
    if (resp.status === 404) {
      if (currentTaskId === taskId) {
        stopStatusPolling();
        resetPipelineButtons();
        currentTaskId = null;
      }
      return;
    }
    const data = await resp.json();
    updatePipelineStatus(data.status, data.progress || data.error || '');
    syncContinuousSequenceStatus(data);

    if (data.status === 'completed') {
      if (currentTaskId === taskId) {
        await pollPoolStatus(taskId);
        stopStatusPolling();
        disconnectStreamWs();
        currentTaskId = null;
        const sequence = continuousMonitorState;
        if (sequence && sequence.currentTaskId === taskId) {
          (data.playlist_results || []).forEach(result => {
            if (result?.filename) markSequenceVideo(result.filename, result.status || 'completed');
          });
          sequence.currentTaskId = null;
          finishContinuousMonitoring(data.progress || 'Continuous monitoring completed', 'completed');
          return;
        }
        resetPipelineButtons();
        showToast('✅ 处理完成!');
        const resultPlaceholder = document.getElementById('resultPlaceholder');
        if (resultPlaceholder) {
          resultPlaceholder.innerHTML = '<span>✅</span><p>处理完成</p>';
          resultPlaceholder.className = 'video-placeholder';
          resultPlaceholder.style.cssText = '';
        }
        loadTaskHistory();
      }
    } else if (data.status === 'failed') {
      if (currentTaskId === taskId) {
        await pollPoolStatus(taskId);
        stopStatusPolling();
        disconnectStreamWs();
        const errorMsg = data.error || '未知错误';
        currentTaskId = null;
        const sequence = continuousMonitorState;
        if (sequence && sequence.currentTaskId === taskId) {
          (data.playlist_results || []).forEach(result => {
            if (result?.filename) markSequenceVideo(result.filename, result.status || 'failed');
          });
          sequence.currentTaskId = null;
          finishContinuousMonitoring(errorMsg === '用户手动停止' ? 'Continuous monitoring stopped' : `Sequence failed: ${errorMsg}`, 'failed');
          return;
        }
        resetPipelineButtons();
        _restoreResultPlaceholder();
        if (errorMsg === '用户手动停止') {
          showToast('已停止', 'info');
        } else {
          showToast('处理失败: ' + errorMsg, 'error');
        }
        loadTaskHistory();
      }
    }
  } catch (e) {
    console.error('状态轮询失败:', e);
  }
}

async function pollPoolStatus(targetTaskId = null) {
  const taskId = targetTaskId || currentTaskId;
  if (!taskId) return;
  try {
    const resp = await fetch(`${PIPE_API}/pool-status/${taskId}`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderPoolRows('candidatePoolRows', data.candidate || [], '等待候选帧');
    renderPoolRows('keyframePoolRows', data.keyframe || [], '等待正式关键帧');
    renderTrackRows(data.track || []);
  } catch (e) {}
}

function pipelineRecordStatus(value) {
  const text = String(value || '-').trim();
  const rules = [
    [/已进入正式池|进入正式池/, 'Promoted'],
    [/正式帧|正式池/, 'Confirmed'],
    [/临时帧|临时池|候选帧|候选池/, 'Candidate'],
    [/正在追踪|追踪中|正在跟踪|跟踪中/, 'Tracking'],
    [/已完成|处理完成|结束/, 'Completed'],
    [/丢失/, 'Lost'],
  ];
  const match = rules.find(([pattern]) => pattern.test(text));
  return match ? match[1] : text;
}

function pipelineMemoryText(value) {
  return String(value || '-')
    .replaceAll('临时帧', 'Candidate')
    .replaceAll('正式帧', 'Confirmed')
    .replaceAll('临时池', 'Candidate Pool')
    .replaceAll('正式池', 'Keyframe Pool');
}

function pipelineProgressText(status, value) {
  const text = String(value || '').trim();
  const exact = {
    '请选择监控视频': 'Select a monitoring video',
    '等待开始': 'Ready to start',
    '处理中...': 'Processing...',
    '处理中…': 'Processing...',
    '处理完成': 'Processing completed',
    '正在停止...': 'Stopping...',
    '已停止': 'Stopped',
  };
  if (exact[text]) return exact[text];
  if (status === 'completed' && (!text || /完成/.test(text))) return 'Processing completed';
  if (status === 'failed' && !text) return 'Processing failed';
  if (status === 'running' && !text) return 'Processing...';
  if (status === 'idle' && !text) return 'Ready to start';
  return text || status;
}

function renderPoolRows(elementId, rows, emptyText) {
  const box = document.getElementById(elementId);
  if (!box) return;
  if (!rows.length) {
    box.innerHTML = `<div class="pool-empty">${escHtml(emptyText)}</div>`;
    return;
  }
  box.innerHTML = rows.slice(0, 40).map(row => {
    const hull = row.hullNumber || 'Unreadable';
    const status = pipelineRecordStatus(row.status);
    return `<div class="pool-table-row">
      <div class="pool-track"><strong>${escHtml(row.trackId || '-')}</strong><span>${escHtml(status)} · ${escHtml(row.time || '-')}</span></div>
      <div class="pool-hull">${escHtml(hull)}</div>
      <div class="pool-description" title="${safeAttr(row.description || '-')}">${escHtml(row.description || '-')}</div>
    </div>`;
  }).join('');
}

function initializeContinuousMonitorControls() {
  updateContinuousMonitorControls();
}

function toggleContinuousVideo(filename, checked) {
  if (continuousMonitorState) return;
  if (checked) continuousSelectedVideos.add(filename);
  else continuousSelectedVideos.delete(filename);
  const item = [...document.querySelectorAll('.video-item')].find(node => node.dataset.name === filename);
  if (item) item.classList.toggle('sequence-selected', checked);
  const stateLabel = [...document.querySelectorAll('.video-sequence-state')].find(node => node.dataset.sequenceState === filename);
  if (stateLabel) stateLabel.textContent = checked ? 'Queued' : 'Not Selected';
  updateContinuousMonitorControls();
}

function toggleAllContinuousVideos(checked) {
  if (continuousMonitorState) return;
  const visibleVideos = getVisibleVideoCatalog();
  const visibleNames = new Set(visibleVideos.map(video => video.filename));
  if (checked) {
    visibleNames.forEach(filename => continuousSelectedVideos.add(filename));
  } else {
    visibleNames.forEach(filename => continuousSelectedVideos.delete(filename));
  }
  document.querySelectorAll('.video-sequence-check input').forEach(input => { input.checked = checked; });
  document.querySelectorAll('.video-item').forEach(item => item.classList.toggle('sequence-selected', checked));
  document.querySelectorAll('.video-sequence-state').forEach(label => { label.textContent = checked ? 'Queued' : 'Not Selected'; });
  updateContinuousMonitorControls();
}

function updateContinuousMonitorControls() {
  const selectedCount = document.getElementById('continuousSelectedCount');
  const selectAll = document.getElementById('continuousSelectAll');
  const startButton = document.getElementById('btnStartContinuous');
  const stopButton = document.getElementById('btnStopContinuous');
  const active = Boolean(continuousMonitorState);
  const visibleVideos = getVisibleVideoCatalog();
  const visibleNames = new Set(visibleVideos.map(video => video.filename));
  const count = visibleVideos.filter(video => continuousSelectedVideos.has(video.filename)).length;
  if (selectedCount) selectedCount.textContent = `${count} Selected`;
  if (selectAll) {
    selectAll.checked = visibleVideos.length > 0 && count === visibleVideos.length;
    selectAll.indeterminate = count > 0 && count < visibleVideos.length;
    selectAll.disabled = active || visibleVideos.length === 0;
  }
  const singleStartButton = document.getElementById('btnStartPipeline');
  if (singleStartButton) singleStartButton.disabled = active;
  if (startButton) {
    startButton.disabled = active || count === 0;
    startButton.style.display = active ? 'none' : '';
  }
  if (stopButton) {
    stopButton.style.display = active ? '' : 'none';
    stopButton.disabled = false;
  }
  document.querySelectorAll('.video-sequence-check input').forEach(input => {
    input.disabled = active || !visibleNames.has(input.dataset.name);
  });
  ['continuousStartTime', 'continuousGapSeconds', 'continuousFailurePolicy'].forEach(id => {
    const element = document.getElementById(id);
    if (element) element.disabled = active;
  });
}

function updateContinuousProgress(text) {
  const element = document.getElementById('continuousProgress');
  if (!element) return;
  element.textContent = text;
  const status = element.closest('.continuous-monitor-status');
  if (status) status.classList.toggle('is-active', Boolean(continuousMonitorState));
}

function markSequenceVideo(filename, state) {
  const target = [...document.querySelectorAll('.video-item')].find(item => item.dataset.name === filename);
  if (state === 'running') {
    document.querySelectorAll('.video-item.sequence-running').forEach(item => {
      if (item !== target) item.classList.remove('sequence-running');
    });
  }
  if (target) {
    target.classList.remove('sequence-running', 'sequence-completed', 'sequence-failed', 'sequence-skipped');
    if (state !== 'queued') target.classList.add(`sequence-${state}`);
  }
  const stateLabels = {running: 'Running', completed: 'Completed', failed: 'Failed', skipped: 'Skipped', queued: 'Queued'};
  const label = [...document.querySelectorAll('.video-sequence-state')].find(node => node.dataset.sequenceState === filename);
  if (label) label.textContent = stateLabels[state] || 'Queued';
}

function renderActiveTrackMonitor(rows) {
  const container = document.getElementById('activeTrackMonitorRows');
  if (!container) return;
  const activeRows = Array.isArray(rows) ? rows : [];
  const previousScrollTop = container.scrollTop;
  const previousScrollHeight = container.scrollHeight;
  const followsLatest = previousScrollTop < 12;
  if (!activeRows.length) {
    container.innerHTML = '<div class="active-track-monitor-empty"><strong>Awaiting trajectory observations...</strong><small>Time · Status · Track<br>Visual description</small></div>';
    return;
  }
  container.innerHTML = activeRows.slice(0, 40).map((row) => {
    const trackId = row.trackId || '-';
    const status = pipelineRecordStatus(row.status);
    const hull = row.hullNumber || 'Unreadable';
    const description = row.description || 'No visual description';
    const descriptionText = hull === 'Unreadable' ? description : `${hull} · ${description}`;
    return `<article class="active-track-live-row">
      <div class="active-track-live-line"><time>${escHtml(row.time || '--:--:--')}</time><i aria-hidden="true">·</i><span>${escHtml(status)}</span><i aria-hidden="true">·</i><strong>Track #${escHtml(trackId)}</strong></div>
      <p><span class="active-track-description-label">Description:</span><b>${escHtml(descriptionText)}</b></p>
    </article>`;
  }).join('');
  if (followsLatest) {
    container.scrollTop = 0;
  } else {
    container.scrollTop = previousScrollTop + Math.max(0, container.scrollHeight - previousScrollHeight);
  }
}

function renderTrackRows(rows) {
  renderActiveTrackMonitor(rows);
  const box = document.getElementById('trackStatusRows');
  if (!box) return;
  if (!rows.length) {
    box.innerHTML = '<div class="pool-empty">Waiting for active tracks</div>';
    return;
  }
  box.innerHTML = rows.slice(0, 40).map(row => {
    const hull = row.hullNumber || 'Unreadable';
    const description = row.description || '-';
    const status = pipelineRecordStatus(row.status);
    const memoryInfo = pipelineMemoryText(row.memoryInfo);
    return `<div class="pool-table-row track-status-row">
      <div class="pool-track"><strong>${escHtml(row.trackId || '-')}</strong><span>Last observed · ${escHtml(row.time || '-')}</span></div>
      <div class="pool-track-state">${escHtml(status)}</div>
      <div class="pool-track-memory"><strong>${escHtml(memoryInfo)}</strong><span title="${safeAttr(`${hull} · ${description}`)}">${escHtml(hull)} · ${escHtml(description)}</span></div>
    </div>`;
  }).join('');
}

function clearPoolTables() {
  renderPoolRows('candidatePoolRows', [], 'Waiting for candidate frames');
  renderPoolRows('keyframePoolRows', [], 'Waiting for confirmed keyframes');
  renderTrackRows([]);
}

function updatePipelineStatus(status, text) {
  const dot = document.querySelector('#pipelineStatus .status-dot');
  const statusText = document.getElementById('pipelineStatusText');
  if (!dot || !statusText) return;
  dot.className = 'status-dot ' + (status === 'running' ? 'running' : status === 'completed' ? 'completed' : status === 'failed' ? 'failed' : 'idle');
  statusText.textContent = pipelineProgressText(status, text);
}

function resetPipelineStatus() {
  updatePipelineStatus('idle', 'Ready to start');
  resetPipelineButtons();
}

function resetPipelineButtons() {
  const startBtn = document.getElementById('btnStartPipeline');
  const stopBtn = document.getElementById('btnStopPipeline');
  if (startBtn) { startBtn.style.display = ''; startBtn.disabled = false; startBtn.innerHTML = '▶ Start Processing'; }
  if (stopBtn) stopBtn.style.display = 'none';
}

/** 恢复结果区域为初始占位状态 */
function _restoreResultPlaceholder() {
  if (selectedVideo) {
    showVideoPreview(selectedVideo);
  }
  const resultPlaceholder = document.getElementById('resultPlaceholder');
  if (resultPlaceholder && !selectedVideo) {
    resultPlaceholder.innerHTML = '<span>🎬</span><p>选择视频后显示预览</p>';
    resultPlaceholder.className = 'video-placeholder';
    resultPlaceholder.style.cssText = '';
  }
  const logBox = document.getElementById('pipelineLogBox');
  if (logBox) logBox.style.display = 'none';
}

// ── 任务历史 ──
async function loadTaskHistory() {
  const container = document.getElementById('taskHistory');
  if (!container) return;
  try {
    const resp = await fetch(`${PIPE_API}/status`);
    const data = await resp.json();
    if (!data.tasks.length) {
      container.innerHTML = '<div class="empty-msg">暂无任务</div>';
      return;
    }
    container.innerHTML = data.tasks.map(t => {
      const statusIcon = t.status === 'completed' ? '✅' : t.status === 'running' ? '⏳' : '❌';
      const statusClass = t.status === 'completed' ? 'success' : t.status === 'running' ? 'running' : 'error';
      const cameraTag = t.is_camera ? ' <span style="color:#f57c00;font-size:12px">[摄像头]</span>' : '';
      return `
        <div class="task-item ${statusClass}">
          <div class="task-icon">${statusIcon}</div>
          <div class="task-info">
            <div class="task-name">${escHtml(t.video_filename)}${cameraTag}</div>
            <div class="task-meta">
              任务 ${escHtml(t.task_id)} · ${escHtml(t.progress || t.error || t.status)}
            </div>
          </div>
          <div class="task-actions">
            ${t.status === 'running' ? `<button class="btn btn-danger btn-sm" onclick="stopTaskById(this.dataset.id)" data-id="${safeAttr(t.task_id)}">⏹ 停止</button>` : ''}
          </div>
        </div>
      `;
    }).join('');
  } catch (e) {
    container.innerHTML = `<div class="empty-msg">加载失败: ${e.message}</div>`;
  }
}

async function clearTaskHistory() {
  try {
    const resp = await fetch(`${PIPE_API}/tasks/clear`, { method: 'DELETE' });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '清空失败');
    showToast(data.message || '已清空');
    loadTaskHistory();
  } catch (e) {
    showToast('清空失败: ' + e.message, 'error');
  }
}

async function stopTaskById(taskId) {
  try {
    await fetch(`${PIPE_API}/stop/${taskId}`, { method: 'POST' });
    showToast('已停止');
    loadTaskHistory();
  } catch (e) {
    showToast(e.message, 'error');
  }
}

// ═══════════════════════════════════════════
// 摄像头 Demo
// ═══════════════════════════════════════════

let cameraTaskId = null;
let cameraPollTimer = null;
let browserCameraStream = null;   // MediaStream
let browserCameraWs = null;       // WebSocket
let browserCameraTimer = null;    // 帧捕获定时器
let browserCameraCanvas = null;   // 离屏 canvas
let browserCameraMediaRecorder = null; // H264 MediaRecorder
let browserCameraCaptureFps = 15; // 推帧帧率

function onCameraSourceChange() {
  const sel = document.getElementById('cameraSource');
  if (!sel) return;
  const val = sel.value;
  const urlInput = document.getElementById('cameraUrl');
  const previewRow = document.getElementById('browserCameraPreviewRow');
  const streamModeRow = document.getElementById('camStreamModeRow');
  const streamModeHint = document.getElementById('camStreamModeHint');

  if (urlInput) {
    urlInput.style.display = (val === '0' || val === 'browser') ? 'none' : '';
    if (val === 'rtsp') {
      urlInput.placeholder = 'rtsp://192.168.1.100/stream';
    } else if (val === 'custom') {
      urlInput.placeholder = '输入视频路径或 URL';
    }
  }

  if (previewRow) {
    previewRow.style.display = val === 'browser' ? '' : 'none';
  }

  // H264/MJPEG 切换仅对浏览器摄像头可见；非浏览器时显示提示
  const isBrowser = val === 'browser';
  if (streamModeRow) streamModeRow.style.display = isBrowser ? '' : 'none';
  if (streamModeHint) streamModeHint.style.display = isBrowser ? 'none' : '';
}

function getCameraInput() {
  const sel = document.getElementById('cameraSource');
  if (!sel) return '';
  if (sel.value === '0') return '0';
  if (sel.value === 'browser') return '__browser__';
  const urlInput = document.getElementById('cameraUrl');
  return urlInput ? urlInput.value.trim() : '';
}

// ── 浏览器摄像头：启动 ──
async function startBrowserCamera() {
  const btn = document.getElementById('btnStartCamera');
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span> 启动中...';

  const streamMode = (document.getElementById('camStreamMode') || {}).value || 'mjpeg';

  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('当前页面不是安全上下文（需要 HTTPS 或 localhost），浏览器不允许访问摄像头');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      audio: false,
    }).catch(err => {
      if (err.name === 'NotAllowedError') throw new Error('摄像头权限被拒绝，请在浏览器弹窗中点击"允许"');
      if (err.name === 'NotFoundError') throw new Error('未检测到摄像头设备，请确认电脑有可用摄像头');
      if (err.name === 'NotReadableError') throw new Error('摄像头被其他程序占用，请关闭其他使用摄像头的应用');
      throw new Error('摄像头访问失败: ' + err.message);
    });
    browserCameraStream = stream;

    const preview = document.getElementById('browserCameraPreview');
    const placeholder = document.getElementById('browserCameraPreviewPlaceholder');
    if (preview) {
      preview.srcObject = stream;
      preview.style.display = '';
    }
    if (placeholder) placeholder.style.display = 'none';

    const params = collectCameraParams();
    const resp = await fetch(`${PIPE_API}/start-browser-camera`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        stream_mode: streamMode,
        ...params,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '启动失败');

    cameraTaskId = data.task_id;
    browserCameraCaptureFps = data.capture_fps || 15;

    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProto}://${location.host}${PIPE_API}/ws/camera/${cameraTaskId}`;

    if (streamMode === 'webrtc') {
      // ── WebRTC 模式：浏览器直连服务器，超低延迟 ──
      setupWebRTCCamera(cameraTaskId, stream);
    } else if (streamMode === 'h264') {
      // ── H264 模式：MediaRecorder 编码 → WebSocket ──
      setupH264CameraWs(wsUrl, stream);
    } else {
      // ── MJPEG 模式：逐帧 JPEG → WebSocket ──
      setupMjpegCameraWs(wsUrl, stream);
    }

  } catch (e) {
    showToast('启动失败: ' + e.message, 'error');
    stopBrowserCamera();
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ 启动摄像头识别';
  }
}

/** MJPEG 模式：逐帧 JPEG 推流 */
function setupMjpegCameraWs(wsUrl, stream) {
  function setupWsHandlers(ws) {
    ws.onopen = () => {
      showToast('摄像头已连接 (MJPEG)，开始推流');
      updateCameraStatus('running', 'MJPEG 推流中...');
      document.getElementById('btnStartCamera').style.display = 'none';
      document.getElementById('btnStopCamera').style.display = '';

      connectCameraH264(cameraTaskId);
      startFrameCapture(ws, stream);
      startCameraPolling();
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (!msg.ok) console.warn('帧发送失败:', msg.error);
      } catch {}
    };

    ws.onerror = () => { console.warn('MJPEG WebSocket 错误'); };

    ws.onclose = (evt) => {
      if (!cameraTaskId) return;
      if (evt.code !== 1000) {
        showToast('摄像头连接断开，尝试重连…', 'info');
        _scheduleReconnect('mjpeg-cam', () => {
          if (!cameraTaskId) return;
          const newWs = new WebSocket(wsUrl);
          browserCameraWs = newWs;
          setupWsHandlers(newWs);
        }, cameraTaskId);
      }
    };
  }

  const ws = new WebSocket(wsUrl);
  browserCameraWs = ws;
  setupWsHandlers(ws);
}

/** H264 模式：MediaRecorder 编码 → WebSocket 推流 */
function setupH264CameraWs(wsUrl, stream) {
  // 检查 H264 MediaRecorder 支持 — 优先 avc1，fallback vp8
  const h264Mimes = [
    'video/mp4; codecs="avc1.42E01E"',  // Baseline 3.1
    'video/mp4; codecs="avc1.4D401E"',  // Main 3.1
    'video/mp4; codecs="avc1.64001E"',  // High 3.1
    'video/webm; codecs="h264"',        // WebM 容器 + H264
  ];
  const vp8Mime = 'video/webm; codecs="vp8"';

  let useMime = null;
  let codecName = null;

  for (const mime of h264Mimes) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mime)) {
      useMime = mime;
      codecName = 'h264';
      break;
    }
  }
  if (!useMime && typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(vp8Mime)) {
    useMime = vp8Mime;
    codecName = 'vp8';
  }

  if (!useMime) {
    showToast('当前浏览器不支持 MediaRecorder 编码，已回退到 MJPEG 模式', 'info');
    setupMjpegCameraWs(wsUrl, stream);
    return;
  }

  console.log(`[H264 Camera] 使用 codec: ${codecName}, mime: ${useMime}`);
  showToast(`使用编码: ${useMime}`, 'info');

  const ws = new WebSocket(wsUrl);
  browserCameraWs = ws;

  ws.onopen = () => {
    showToast(`摄像头已连接 (${codecName.toUpperCase()})，开始推流`);
    updateCameraStatus('running', `${codecName.toUpperCase()} 推流中...`);
    document.getElementById('btnStartCamera').style.display = 'none';
    document.getElementById('btnStopCamera').style.display = '';

    // 结果推流：H264 MSE 播放
    connectCameraH264(cameraTaskId);
    startCameraPolling();

    // 首条消息：JSON 文本告知后端编码格式
    ws.send(JSON.stringify({ codec: codecName }));

    // 创建 MediaRecorder
    try {
      const recorder = new MediaRecorder(stream, {
        mimeType: useMime,
        videoBitsPerSecond: 1_500_000, // 1.5 Mbps（降低码率减少编码延迟）
      });
      browserCameraMediaRecorder = recorder;

      let chunkCount = 0;
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          chunkCount++;
          e.data.arrayBuffer().then(buf => ws.send(buf));
        }
      };
      recorder.onerror = (e) => {
        console.error('MediaRecorder 错误:', e.error);
        showToast('编码出错，已停止推流', 'error');
      };
      // timeslice=100ms: 产出频率约 10 chunk/s，平衡延迟和解码稳定性
      recorder.start(100);
      console.log(`[H264 Camera] MediaRecorder 已启动, timeslice=100ms`);
    } catch (e) {
      console.error('MediaRecorder 创建失败:', e);
      showToast('编码启动失败: ' + e.message, 'error');
      ws.close();
    }
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (!msg.ok) console.warn('帧处理失败:', msg.error);
    } catch {}
  };

  ws.onerror = () => { console.warn('H264 WebSocket 错误'); };

  ws.onclose = (evt) => {
    if (!cameraTaskId) return;
    if (evt.code !== 1000) {
      showToast('摄像头连接断开，尝试重连…', 'info');
      _scheduleReconnect('h264-upload-cam', () => {
        if (!cameraTaskId) return;
        const newWs = new WebSocket(wsUrl);
        browserCameraWs = newWs;
        // 重连时重新走完整 onopen 流程（简化处理：回退 MJPEG）
        showToast('H264 重连暂不支持，已回退 MJPEG', 'info');
        setupMjpegCameraWs(wsUrl, stream);
      }, cameraTaskId);
    }
  };
}

/** WebRTC 模式：浏览器直连服务器，超低延迟推流（连接超时自动降级到 H264 WebSocket） */
function setupWebRTCCamera(taskId, stream) {
  let pc = null;
  let webrtcConnected = false;

  async function connect() {
    try {
      pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'turn:218.106.147.53:3478', username: 'webrtc', credential: '123456' },
        ],
      });

      // 添加摄像头轨道
      stream.getTracks().forEach(track => pc.addTrack(track, stream));

      // ICE 候选收集：优先等 srflx（STUN 公网）候选，但 gathering 完成即发 offer（服务端会补 srflx）
      const offer = await pc.createOffer({
        offerToReceiveVideo: true,
        offerToReceiveAudio: false,
      });
      await pc.setLocalDescription(offer);

      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
          return;
        }
        let hasSrflx = false;
        const timer = setTimeout(() => {
          resolve();
        }, 8000);
        pc.addEventListener('icecandidate', (e) => {
          if (e.candidate && e.candidate.type === 'srflx') {
            hasSrflx = true;
          }
        });
        pc.addEventListener('icegatheringstatechange', () => {
          if (pc.iceGatheringState === 'complete') {
            clearTimeout(timer);
            resolve();
          }
        });
      });

      // 发送 offer 给服务器
      const resp = await fetch(`${PIPE_API}/webrtc/offer/${taskId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'WebRTC 信令失败');
      }

      const answer = await resp.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer));

      // Trickle ICE：offer 发出后 STUN 公网候选才到，补发给服务端
      pc.addEventListener('icecandidate', (e) => {
        if (!e.candidate) return;
        fetch(`${PIPE_API}/webrtc/candidate/${taskId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            candidate: e.candidate.candidate,
            sdpMid: e.candidate.sdpMid,
            sdpMLineIndex: e.candidate.sdpMLineIndex,
          }),
        }).catch(() => {});
      });

      // 等待 WebRTC 连接建立，超时自动降级
      await new Promise((resolve, reject) => {
        const FALLBACK_TIMEOUT = 10000;
        const timer = setTimeout(() => {
          if (!webrtcConnected) {
            reject(new Error('timeout'));
          }
        }, FALLBACK_TIMEOUT);
        pc.addEventListener('connectionstatechange', () => {
          if (pc.connectionState === 'connected') {
            webrtcConnected = true;
            clearTimeout(timer);
            resolve();
          } else if (pc.connectionState === 'failed') {
            clearTimeout(timer);
            reject(new Error('failed'));
          }
        });
        // 已经 connected 的情况
        if (pc.connectionState === 'connected') {
          webrtcConnected = true;
          clearTimeout(timer);
          resolve();
        }
      });

      showToast('摄像头已连接 (WebRTC)，开始推流');
      updateCameraStatus('running', 'WebRTC 推流中...');
      document.getElementById('btnStartCamera').style.display = 'none';
      document.getElementById('btnStopCamera').style.display = '';

      connectCameraH264(taskId);
      startCameraPolling();

      browserCameraTimer = true;

    } catch (e) {
      // WebRTC 失败或超时，自动降级到 H264 WebSocket
      console.warn('WebRTC 连接失败，降级到 H264 WebSocket:', e.message);
      if (pc) { pc.close(); pc = null; }

      showToast('WebRTC 不可用，自动切换到 H264 WebSocket 模式', 'info');
      const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
      const wsUrl = `${wsProto}://${location.host}${PIPE_API}/ws/camera/${taskId}`;
      setupH264CameraWs(wsUrl, stream);
    }
  }

  connect();

  // 暴露给 stopCameraPipeline 使用
  browserCameraWs = { close: () => { if (pc) { pc.close(); pc = null; } } };
}

function startFrameCapture(ws, stream) {
  const video = document.getElementById('browserCameraPreview');
  if (!video) return;

  const OUT_W = 640;
  const OUT_H = 480;
  const CAPTURE_INTERVAL = Math.round(1000 / (browserCameraCaptureFps || 15));
  const JPEG_QUALITY = 0.7;

  const doCapture = () => {
    if (!browserCameraCanvas) {
      browserCameraCanvas = document.createElement('canvas');
    }
    const canvas = browserCameraCanvas;
    canvas.width = OUT_W;
    canvas.height = OUT_H;
    const ctx = canvas.getContext('2d');

    const capture = () => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(video, 0, 0, OUT_W, OUT_H);

      // toBlob 异步但比 toDataURL 轻量，避免主线程 base64 编解码开销
      canvas.toBlob((blob) => {
        if (!blob || ws.readyState !== WebSocket.OPEN) return;
        const reader = new FileReader();
        reader.onload = () => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(reader.result);
          }
        };
        reader.readAsArrayBuffer(blob);
      }, 'image/jpeg', JPEG_QUALITY);
    };

    browserCameraTimer = setInterval(capture, CAPTURE_INTERVAL);
  };

  if (video.readyState >= 2) {
    doCapture();
  } else {
    video.addEventListener('loadeddata', doCapture, { once: true });
  }
}

function stopFrameCapture() {
  _clearReconnect('mjpeg-cam');
  _clearReconnect('h264-upload-cam');
  if (browserCameraMediaRecorder && browserCameraMediaRecorder.state !== 'inactive') {
    browserCameraMediaRecorder.stop();
  }
  browserCameraMediaRecorder = null;
  if (browserCameraTimer) {
    if (typeof browserCameraTimer === 'number') {
      clearInterval(browserCameraTimer);
    }
    browserCameraTimer = null;
  }
  if (browserCameraWs) {
    browserCameraWs.close();
    browserCameraWs = null;
  }
  if (browserCameraStream) {
    browserCameraStream.getTracks().forEach(t => t.stop());
    browserCameraStream = null;
  }
  const preview = document.getElementById('browserCameraPreview');
  if (preview) {
    preview.srcObject = null;
    preview.style.display = 'none';
  }
  const placeholder = document.getElementById('browserCameraPreviewPlaceholder');
  if (placeholder) placeholder.style.display = '';
  browserCameraCanvas = null;
}

async function startCameraPipeline() {
  const input = getCameraInput();

  if (input === '__browser__') {
    await startBrowserCamera();
    return;
  }

  if (!input) { showToast('请输入摄像头地址', 'error'); return; }

  const btn = document.getElementById('btnStartCamera');
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span> 启动中...';

  try {
    let videoFilename;
    if (input === '0') {
      videoFilename = '__camera__0';
    } else if (input.startsWith('rtsp://') || input.startsWith('rtmp://') || input.startsWith('http://')) {
      videoFilename = input;
    } else {
      videoFilename = input;
    }

    const resp = await fetch(`${PIPE_API}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_filename: videoFilename,
        ...collectCameraParams(),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '启动失败');

    cameraTaskId = data.task_id;
    updateCameraStatus('running', '摄像头识别运行中...');
    document.getElementById('btnStartCamera').style.display = 'none';
    document.getElementById('btnStopCamera').style.display = '';
    showToast('摄像头 Pipeline 已启动');

    // H.264 WebSocket + MSE 播放
    connectCameraH264(cameraTaskId);

    startCameraPolling();
  } catch (e) {
    showToast('启动失败: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ 启动摄像头识别';
  }
}

async function stopCameraPipeline() {
  const taskId = cameraTaskId;

  // 立即停止轮询，防止后续 pollCameraStatus 干扰新任务
  stopCameraPolling();
  cameraTaskId = null;

  stopFrameCapture();

  // 断开 H.264 推流
  disconnectCameraH264();

  updateCameraStatus('idle', '正在停止...');
  resetCameraButtons();

  if (taskId) {
    try {
      await fetch(`${PIPE_API}/stop/${taskId}`, { method: 'POST' });
    } catch {}
  }

  const cameraStream = document.getElementById('cameraStream');
  const cameraPlaceholder = document.getElementById('cameraStreamPlaceholder');
  if (cameraStream) {
    cameraStream.pause();
    cameraStream.src = '';
    cameraStream.style.display = 'none';
  }
  if (cameraPlaceholder) cameraPlaceholder.style.display = '';

  showToast('摄像头已停止');
}

function startCameraPolling() {
  stopCameraPolling();
  cameraPollTimer = setInterval(pollCameraStatus, 3000);
}

function stopCameraPolling() {
  if (cameraPollTimer) {
    clearInterval(cameraPollTimer);
    cameraPollTimer = null;
  }
}

async function pollCameraStatus() {
  // 快照当前任务 ID，防止请求返回时 cameraTaskId 已变为新任务
  const taskId = cameraTaskId;
  if (!taskId) return;
  try {
    const resp = await fetch(`${PIPE_API}/status/${taskId}`);
    if (resp.status === 404) {
      if (cameraTaskId === taskId) {
        stopCameraPolling();
        resetCameraButtons();
        cameraTaskId = null;
      }
      return;
    }
    const data = await resp.json();
    updateCameraStatus(data.status, data.progress || data.error || '');

    if (data.status !== 'running') {
      if (cameraTaskId === taskId) {
        stopCameraPolling();
        resetCameraButtons();
        disconnectCameraH264();
        if (data.status === 'completed') {
          showToast('✅ 摄像头处理完成');
        } else if (data.status === 'failed') {
          const errorMsg = data.error || '未知错误';
          if (errorMsg === '用户手动停止') {
            showToast('摄像头已停止', 'info');
          } else {
            showToast('摄像头处理失败: ' + errorMsg, 'error');
          }
        }
        cameraTaskId = null;
      }
    }
  } catch (e) {
    console.error('摄像头状态轮询失败:', e);
  }
}

function updateCameraStatus(status, text) {
  const dot = document.querySelector('#cameraStatus .status-dot');
  const statusText = document.getElementById('cameraStatusText');
  if (!dot || !statusText) return;
  dot.className = 'status-dot ' + (status === 'running' ? 'running' : status === 'completed' ? 'completed' : status === 'failed' ? 'failed' : 'idle');
  statusText.textContent = text || status;
}

function resetCameraButtons() {
  const startBtn = document.getElementById('btnStartCamera');
  const stopBtn = document.getElementById('btnStopCamera');
  if (startBtn) startBtn.style.display = '';
  if (stopBtn) stopBtn.style.display = 'none';
}

// ── 摄像头 H.264 推流状态 ──
let _camH264Ws = null;
let _camH264MediaSource = null;
let _camH264SourceBuffer = null;
let _camH264ObjectUrl = null;
let _camH264Queue = [];

function connectCameraH264(taskId) {
  disconnectCameraH264();

  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${wsProto}://${location.host}${PIPE_API}/ws/h264/${taskId}`;

  const videoEl = document.getElementById('cameraStream');
  const placeholder = document.getElementById('cameraStreamPlaceholder');
  const fpsEl = document.getElementById('cameraStreamFps');
  if (!videoEl) return;

  // 显示 video，隐藏 placeholder
  videoEl.style.display = '';
  if (placeholder) placeholder.style.display = 'none';
  if (fpsEl) { fpsEl.style.display = ''; fpsEl.textContent = '正在连接推流...'; }

  const ms = new MediaSource();
  _camH264ObjectUrl = URL.createObjectURL(ms);
  videoEl.src = _camH264ObjectUrl;
  videoEl.load();  // 强制加载，确保 sourceopen 触发
  _camH264MediaSource = ms;
  _camH264SourceBuffer = null;
  _camH264Queue = [];

  let cameraSegmentRate = null;

  function _updateCameraStreamStatus() {
    if (!fpsEl) return;
    const resolution = videoEl.videoWidth > 0 ? `${videoEl.videoWidth}×${videoEl.videoHeight}` : '读取中';
    const rate = cameraSegmentRate === null ? '' : ` · 更新 ${cameraSegmentRate} 次/秒`;
    fpsEl.textContent = `推流 ${resolution}${rate}`;
  }

  videoEl.addEventListener('loadedmetadata', _updateCameraStreamStatus);
  videoEl.addEventListener('resize', _updateCameraStreamStatus);

  function _processCamQueue() {
    const sb = _camH264SourceBuffer;
    if (!sb || sb.updating) return;
    try {
      const vEl = document.getElementById('cameraStream');
      if (vEl && sb.buffered.length > 0) {
        const liveEdge = sb.buffered.end(sb.buffered.length - 1);
        if (liveEdge - vEl.currentTime > 1.0) vEl.currentTime = Math.max(0, liveEdge - 0.15);
      }
      if (vEl && sb.buffered.length > 0 && sb.buffered.start(0) < vEl.currentTime - 8) {
        sb.remove(sb.buffered.start(0), vEl.currentTime - 3);
        return;
      }
    } catch (e) {}
    if (_camH264Queue.length > 0) {
      try { sb.appendBuffer(_camH264Queue.shift()); } catch (e) {
        console.warn('摄像头推流缓冲异常，重新建立解码连接:', e);
        if (_camH264Ws && _camH264Ws.readyState === WebSocket.OPEN) {
          _camH264Ws.close(1013, '解码缓冲积压');
        }
      }
    }
  }

  function _tryConnect() {
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    _camH264Ws = ws;

    let segmentCount = 0;
    let segmentTimer = performance.now();

    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        const view = new DataView(evt.data);
        const msgType = view.getUint8(0);
        const payload = evt.data.slice(5);

        if (msgType === 0x01) {
          // Init segment（仅首次创建 SourceBuffer）
          if (_camH264SourceBuffer) {
            if (ws.readyState === WebSocket.OPEN) ws.close(1012, '编码上下文已更新');
            return;
          }
          try {
            if (ms.readyState !== 'open') {
              console.warn('摄像头 MediaSource 未就绪，忽略 init segment');
              return;
            }
            const sb = ms.addSourceBuffer('video/mp4; codecs="avc1.42C01F"');
            _camH264SourceBuffer = sb;
            sb.addEventListener('updateend', () => { _processCamQueue(); });
            sb.appendBuffer(payload);
          } catch (e) {
            console.error('摄像头 MSE SourceBuffer 创建失败:', e);
          }
        } else if (msgType === 0x02) {
          // Media segment
          const sb = _camH264SourceBuffer;
          if (!sb) return;
          if (sb.updating) {
            if (_camH264Queue.length >= 24) {
              ws.close(1013, '解码队列积压');
              return;
            }
            _camH264Queue.push(payload);
          } else {
            try { sb.appendBuffer(payload); } catch (e) {
              console.warn('摄像头推流分片追加失败，重新建立解码连接:', e);
              if (ws.readyState === WebSocket.OPEN) ws.close(1013, '分片追加失败');
            }
          }
          segmentCount++;
          const now = performance.now();
          if (now - segmentTimer > 1000) {
            cameraSegmentRate = (segmentCount * 1000 / (now - segmentTimer)).toFixed(1);
            _updateCameraStreamStatus();
            segmentCount = 0;
            segmentTimer = now;
          }
        }
      } else {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'done') {
            disconnectCameraH264();
            if (fpsEl) fpsEl.textContent = '处理完成';
          }
        } catch {}
      }
    };

    ws.onclose = () => {
      if (cameraTaskId === taskId) {
        if (fpsEl) fpsEl.textContent = '正在重建解码连接...';
        _scheduleReconnect('h264-cam', () => {
          if (cameraTaskId === taskId) connectCameraH264(taskId);
        }, taskId);
      }
    };
    ws.onerror = () => {};
  }

  ms.addEventListener('sourceopen', () => {
    if (_camH264MediaSource !== ms) return;
    // 首次延迟 1.5 秒再连接，给后端 ffmpeg 启动时间
    setTimeout(() => {
      if (cameraTaskId === taskId && _camH264MediaSource === ms) _tryConnect();
    }, 1500);
  });
}

function disconnectCameraH264() {
  _clearReconnect('h264-cam');
  if (_camH264Ws) { _camH264Ws.onclose = null; _camH264Ws.close(); _camH264Ws = null; }
  if (_camH264SourceBuffer) {
    try { if (_camH264SourceBuffer.updating) _camH264SourceBuffer.abort(); } catch {}
  }
  if (_camH264MediaSource && _camH264MediaSource.readyState === 'open') {
    try { _camH264MediaSource.endOfStream(); } catch {}
  }
  _camH264MediaSource = null;
  _camH264SourceBuffer = null;
  _camH264Queue = [];
  const videoEl = document.getElementById('cameraStream');
  if (videoEl) { videoEl.pause(); videoEl.removeAttribute('src'); videoEl.load(); }
  if (_camH264ObjectUrl) {
    URL.revokeObjectURL(_camH264ObjectUrl);
    _camH264ObjectUrl = null;
  }
  const fpsEl = document.getElementById('cameraStreamFps');
  if (fpsEl) fpsEl.textContent = '';
}

// ── WebSocket 自动重连（指数退避 + 状态检查 + 最大重试）──
const _reconnectStates = new Map(); // key → {delay, timer, retries}
const MAX_RECONNECT_RETRIES = 5;

async function _checkTaskRunning(taskId) {
  try {
    const resp = await fetch(`${PIPE_API}/status/${taskId}`);
    if (!resp.ok) return false;
    const data = await resp.json();
    return data.status === 'running';
  } catch { return false; }
}

function _scheduleReconnect(key, connectFn, taskId) {
  let state = _reconnectStates.get(key);
  if (!state) {
    state = { delay: 1000, timer: null, retries: 0 };
    _reconnectStates.set(key, state);
  }
  if (state.timer) clearTimeout(state.timer);

  if (state.retries >= MAX_RECONNECT_RETRIES) {
    _reconnectStates.delete(key);
    return;
  }
  state.retries++;

  state.timer = setTimeout(async () => {
    // 重连前检查任务是否还在运行
    if (taskId) {
      const running = await _checkTaskRunning(taskId);
      if (!running) {
        _reconnectStates.delete(key);
        return;
      }
    }
    _reconnectStates.delete(key);
    connectFn();
  }, state.delay);
  state.delay = Math.min(state.delay * 2, 16000); // 1s → 2s → 4s → ... → 16s max
}

function _clearReconnect(key) {
  const state = _reconnectStates.get(key);
  if (state) {
    clearTimeout(state.timer);
    _reconnectStates.delete(key);
  }
}

// ── 工具函数 ──
if (typeof escHtml === 'undefined') {
  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }
}
if (typeof escAttr === 'undefined') {
  function escAttr(s) {
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/'/g, "\\'");
  }
}

/** 安全地将文件名插入 HTML 属性（防 XSS） */
function safeAttr(s) {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

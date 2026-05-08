// ── Realtime Detection ────────────────────────────────────────────────────
let rtStream = null;
let rtTimer  = null;
let rtRunning = false;
let rtFrameW = 640, rtFrameH = 480;
let rtFpsHistory = [];
let rtClassCounts = {longitudinal_crack:0,transverse_crack:0,alligator_crack:0,pothole:0};

const RT_INTERVAL_MS = 150; // ~6-7 fps inference

const CLASS_COLORS = {
  longitudinal_crack: '#f59e0b',
  transverse_crack:   '#ef4444',
  alligator_crack:    '#8b5cf6',
  pothole:            '#06b6d4',
};

async function realtimeStart() {
  const errEl  = document.getElementById('rt-error');
  errEl.classList.add('hidden');
  try {
    rtStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      audio: false,
    });
  } catch (e) {
    errEl.textContent = '❌ Kamera tidak dapat diakses: ' + e.message;
    errEl.classList.remove('hidden');
    return;
  }

  const video = document.getElementById('rt-video');
  video.srcObject = rtStream;
  await video.play();

  document.getElementById('rt-placeholder').classList.add('hidden');
  document.getElementById('rt-hud').classList.remove('hidden');
  document.getElementById('rt-fps-hud').classList.remove('hidden');
  document.getElementById('rt-start-btn').classList.add('hidden');
  document.getElementById('rt-stop-btn').classList.remove('hidden');
  document.getElementById('rt-status-badge').textContent = 'LIVE';
  document.getElementById('rt-status-badge').className = 'badge-online';

  rtRunning = true;
  rtClassCounts = {longitudinal_crack:0,transverse_crack:0,alligator_crack:0,pothole:0};
  rtTimer = setInterval(rtCapture, RT_INTERVAL_MS);
}

function realtimeStop() {
  if (!rtRunning) return;
  rtRunning = false;
  clearInterval(rtTimer);
  if (rtStream) { rtStream.getTracks().forEach(t => t.stop()); rtStream = null; }
  const video = document.getElementById('rt-video');
  video.srcObject = null;
  const canvas = document.getElementById('rt-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('rt-placeholder').classList.remove('hidden');
  document.getElementById('rt-hud').classList.add('hidden');
  document.getElementById('rt-fps-hud').classList.add('hidden');
  document.getElementById('rt-start-btn').classList.remove('hidden');
  document.getElementById('rt-stop-btn').classList.add('hidden');
  document.getElementById('rt-status-badge').textContent = 'OFFLINE';
  document.getElementById('rt-status-badge').className = 'badge-offline';
}

async function rtCapture() {
  if (!rtRunning) return;
  const video  = document.getElementById('rt-video');
  if (video.readyState < 2) return;

  // Offscreen canvas at fixed capture size
  const cap = document.createElement('canvas');
  cap.width  = rtFrameW;
  cap.height = rtFrameH;
  cap.getContext('2d').drawImage(video, 0, 0, rtFrameW, rtFrameH);
  const b64 = cap.toDataURL('image/jpeg', 0.75);

  const t0 = performance.now();
  let resp;
  try {
    const r = await fetch('/predict/frame', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ image: b64 }),
    });
    resp = await r.json();
  } catch { return; }

  const elapsed = performance.now() - t0;
  const fps = Math.round(1000 / elapsed);

  rtFpsHistory.push(fps);
  if (rtFpsHistory.length > 10) rtFpsHistory.shift();
  const avgFps = Math.round(rtFpsHistory.reduce((a,b)=>a+b,0) / rtFpsHistory.length);

  document.getElementById('rt-fps-val').textContent = avgFps;
  document.getElementById('rt-fps-side').textContent = avgFps + ' fps';
  document.getElementById('rt-inf-ms').textContent = (resp.inf_ms || Math.round(elapsed)) + ' ms';
  document.getElementById('stat-fps').textContent = avgFps;

  const dets = resp.detections || [];
  document.getElementById('rt-det-count').textContent = dets.length;

  // Update class counts
  dets.forEach(d => { if (rtClassCounts[d.class] !== undefined) rtClassCounts[d.class]++; });
  Object.keys(rtClassCounts).forEach(cls => {
    const el = document.querySelector(`.breakdown-item[data-class="${cls}"] .count`);
    if (el) el.textContent = rtClassCounts[cls];
  });

  // Detection list
  const detListEl = document.getElementById('rt-det-list');
  if (dets.length === 0) {
    detListEl.innerHTML = '<span class="text-gray-500">Tidak ada kerusakan</span>';
  } else {
    detListEl.innerHTML = dets.map(d =>
      `<div class="flex items-center justify-between py-1">
        <span style="color:${d.color}">${d.emoji} ${d.label}</span>
        <span class="font-mono text-xs text-gray-400">${(d.confidence*100).toFixed(0)}%</span>
      </div>`
    ).join('');
  }

  // Draw boxes on overlay canvas
  rtDrawBoxes(video, dets, resp.width || rtFrameW, resp.height || rtFrameH);
}

function rtDrawBoxes(video, dets, srcW, srcH) {
  const canvas = document.getElementById('rt-canvas');
  const dispW  = video.clientWidth  || video.offsetWidth;
  const dispH  = video.clientHeight || video.offsetHeight;

  // Compute letterbox offsets (object-contain)
  const videoAR = srcW / srcH;
  const contAR  = dispW / dispH;
  let renderW, renderH, offX, offY;
  if (videoAR > contAR) {
    renderW = dispW; renderH = dispW / videoAR;
    offX = 0; offY = (dispH - renderH) / 2;
  } else {
    renderH = dispH; renderW = dispH * videoAR;
    offX = (dispW - renderW) / 2; offY = 0;
  }

  canvas.width  = dispW;
  canvas.height = dispH;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, dispW, dispH);

  const scaleX = renderW / srcW;
  const scaleY = renderH / srcH;

  dets.forEach(det => {
    const [bx1, by1, bx2, by2] = det.bbox;
    const x1 = offX + bx1 * scaleX;
    const y1 = offY + by1 * scaleY;
    const x2 = offX + bx2 * scaleX;
    const y2 = offY + by2 * scaleY;
    const color = det.color || '#00d4ff';

    // Fill
    ctx.fillStyle = color + '22';
    ctx.fillRect(x1, y1, x2-x1, y2-y1);

    // Border
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // Corner accents
    const cl = Math.min(20, (x2-x1)*0.15);
    ctx.lineWidth = 3;
    [[x1,y1,1,1],[x2,y1,-1,1],[x1,y2,1,-1],[x2,y2,-1,-1]].forEach(([cx,cy,dx,dy])=>{
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+dx*cl,cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx,cy+dy*cl); ctx.stroke();
    });

    // Label
    const label = `${det.label} ${(det.confidence*100).toFixed(0)}%`;
    ctx.font = 'bold 12px Inter,sans-serif';
    const tw = ctx.measureText(label).width;
    const th = 16;
    const ty = Math.max(th+4, y1);
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty-th-4, tw+10, th+6);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1+5, ty-2);
  });
}

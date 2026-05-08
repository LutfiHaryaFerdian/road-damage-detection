// ── Video Upload Detection ────────────────────────────────────────────────
let vidFile = null;
let vidSSE  = null;

function vidDragOver(e)  { e.preventDefault(); document.getElementById('vid-drop-zone').classList.add('drag-over'); }
function vidDragLeave(e) { document.getElementById('vid-drop-zone').classList.remove('drag-over'); }
function vidDrop(e) {
  e.preventDefault(); vidDragLeave(e);
  const f = e.dataTransfer.files[0];
  if (f) vidLoadFile(f);
}
function vidFileSelected(e) { if (e.target.files[0]) vidLoadFile(e.target.files[0]); }

function vidLoadFile(f) {
  const ok = ['video/mp4','video/quicktime','video/avi','video/x-msvideo','video/x-matroska'];
  if (!ok.includes(f.type) && !f.name.match(/\.(mp4|mov|avi|mkv)$/i)) {
    alert('Format tidak didukung. Gunakan MP4/MOV/AVI'); return;
  }
  vidFile = f;
  document.getElementById('vid-file-name').textContent = f.name;
  document.getElementById('vid-file-size').textContent = (f.size/1024/1024).toFixed(1) + ' MB';
  document.getElementById('vid-file-info').classList.remove('hidden');
  document.getElementById('vid-drop-zone').classList.add('hidden');
  const btn = document.getElementById('vid-detect-btn');
  btn.disabled = false; btn.classList.remove('opacity-50','cursor-not-allowed');
  btn.textContent = '▶ Proses Video';
  document.getElementById('vid-result-wrap').classList.add('hidden');
}

function vidReset() {
  vidFile = null;
  document.getElementById('vid-file-input').value = '';
  document.getElementById('vid-file-info').classList.add('hidden');
  document.getElementById('vid-drop-zone').classList.remove('hidden');
  const btn = document.getElementById('vid-detect-btn');
  btn.disabled = true; btn.classList.add('opacity-50','cursor-not-allowed');
  btn.textContent = 'Proses Video';
  document.getElementById('vid-progress-wrap').classList.add('hidden');
  document.getElementById('vid-result-wrap').classList.add('hidden');
}

async function vidDetect() {
  if (!vidFile) return;
  const btn = document.getElementById('vid-detect-btn');
  btn.disabled = true; btn.classList.add('opacity-50','cursor-not-allowed');
  btn.textContent = 'Mengirim...';

  document.getElementById('vid-progress-wrap').classList.remove('hidden');
  document.getElementById('vid-result-wrap').classList.add('hidden');
  vidSetProgress(0, 'Mengirim file ke server...');

  const fd = new FormData();
  fd.append('file', vidFile);

  let uid;
  try {
    const r = await fetch('/predict/video', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || 'Upload gagal');
    uid = d.uid;
  } catch (e) {
    alert('❌ ' + e.message);
    btn.disabled = false; btn.classList.remove('opacity-50','cursor-not-allowed');
    btn.textContent = '▶ Proses Video';
    return;
  }

  btn.textContent = 'Memproses...';
  vidListenProgress(uid);
}

function vidSetProgress(pct, msg) {
  document.getElementById('vid-progress-bar').style.width = pct + '%';
  document.getElementById('vid-pct-text').textContent = pct + '%';
  document.getElementById('vid-progress-msg').textContent = msg || '';
}

function vidListenProgress(uid) {
  if (vidSSE) { vidSSE.close(); vidSSE = null; }
  vidSSE = new EventSource(`/predict/video/progress/${uid}`);

  vidSSE.onmessage = e => {
    const d = JSON.parse(e.data);
    vidSetProgress(d.percent || 0, d.message || '');

    if (d.status === 'done') {
      vidSSE.close(); vidSSE = null;
      vidShowResult(d);
      const btn = document.getElementById('vid-detect-btn');
      btn.disabled = false; btn.classList.remove('opacity-50','cursor-not-allowed');
      btn.textContent = '▶ Proses Video Lain';
    } else if (d.status === 'error') {
      vidSSE.close(); vidSSE = null;
      alert('❌ Error: ' + d.message);
      const btn = document.getElementById('vid-detect-btn');
      btn.disabled = false; btn.classList.remove('opacity-50','cursor-not-allowed');
      btn.textContent = '▶ Coba Lagi';
    }
  };
  vidSSE.onerror = () => { if (vidSSE) { vidSSE.close(); vidSSE = null; } };
}

function vidShowResult(d) {
  document.getElementById('vid-progress-wrap').classList.add('hidden');
  const wrap = document.getElementById('vid-result-wrap');
  wrap.classList.remove('hidden');

  const player = document.getElementById('vid-result-player');
  player.src = d.video_url + '?t=' + Date.now();
  player.load();

  const dlBtn = document.getElementById('vid-download-btn');
  dlBtn.href = d.video_url;
  dlBtn.download = 'hasil_deteksi_video.mp4';

  const classes = d.class_counts || {};
  const classHtml = Object.entries(classes).map(([k,v])=>`
    <span class="text-xs bg-white/5 px-2 py-1 rounded">${v}× ${k.replace(/_/g,' ')}</span>
  `).join('');

  document.getElementById('vid-result-stats').innerHTML = `
    <div class="stat-mini">
      <div class="text-2xl font-bold text-cyan-400">${d.total_detections || 0}</div>
      <div class="text-xs text-gray-400">Total Deteksi</div>
    </div>
    <div class="stat-mini">
      <div class="text-2xl font-bold text-purple-400">${d.total_frames || 0}</div>
      <div class="text-xs text-gray-400">Total Frame</div>
    </div>
    <div class="stat-mini col-span-1 md:col-span-1">
      <div class="flex flex-wrap gap-1 justify-center">${classHtml || '<span class="text-gray-500 text-xs">-</span>'}</div>
      <div class="text-xs text-gray-400 mt-1">Per Kelas</div>
    </div>
  `;

  const ts = new Date().toLocaleTimeString('id-ID');
  const classArr = Object.entries(classes).map(([k,v])=>`${v}× ${k}`).join(', ');
  addHistory({
    mode: 'video', total: d.total_detections || 0,
    summary: classArr || 'Tidak ada deteksi', time: ts,
  });
}

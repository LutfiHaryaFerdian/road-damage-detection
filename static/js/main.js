// ── Tab Router ────────────────────────────────────────────────────────────
const PAGES = ['home','realtime','image','video','about'];

function showPage(name) {
  PAGES.forEach(p => {
    document.getElementById('page-' + p).classList.add('hidden');
    const tb = document.getElementById('tab-' + p);
    const tm = document.getElementById('tab-' + p + '-m');
    if (tb) tb.classList.remove('active-tab');
    if (tm) tm.classList.remove('active-tab');
  });
  document.getElementById('page-' + name).classList.remove('hidden');
  const tb = document.getElementById('tab-' + name);
  const tm = document.getElementById('tab-' + name + '-m');
  if (tb) tb.classList.add('active-tab');
  if (tm) tm.classList.add('active-tab');
  if (name !== 'realtime') realtimeStop();
}

// ── Detection History (localStorage) ──────────────────────────────────────
const HISTORY_KEY = 'rd_history';
const MAX_HISTORY = 20;

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; }
}

function saveHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)));
}

function addHistory(entry) {
  const h = getHistory();
  h.unshift(entry);
  saveHistory(h);
  renderHistory();
  updateDashboardStats();
}

function renderHistory() {
  const list = document.getElementById('history-list');
  const h = getHistory();
  if (!h.length) {
    list.innerHTML = '<div class="p-6 text-center text-gray-500 text-sm">Belum ada riwayat deteksi</div>';
    return;
  }
  list.innerHTML = h.slice(0, 10).map(item => `
    <div class="history-item">
      <div class="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-lg flex-shrink-0">
        ${item.mode === 'image' ? '🖼️' : item.mode === 'video' ? '🎬' : '📷'}
      </div>
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium">${item.total} objek terdeteksi
          <span class="text-xs text-gray-500 ml-2">${item.mode}</span>
        </div>
        <div class="text-xs text-gray-500 mt-0.5">${item.summary || ''}</div>
      </div>
      <div class="text-xs text-gray-600 flex-shrink-0">${item.time || ''}</div>
    </div>
  `).join('');
}

function updateDashboardStats() {
  const h = getHistory();
  const total = h.reduce((s, i) => s + (i.total || 0), 0);
  document.getElementById('stat-total').textContent = total;
  const confs = h.filter(i => i.avg_conf).map(i => i.avg_conf);
  if (confs.length) {
    const avg = confs.reduce((a, b) => a + b, 0) / confs.length;
    document.getElementById('stat-conf').textContent = (avg * 100).toFixed(0) + '%';
  }
}

function formatTime() {
  return new Date().toLocaleTimeString('id-ID', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

// ── Fetch API Status ───────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.status === 'online') {
      document.getElementById('model-status-badge').textContent = 'YOLOv8 Online';
      document.getElementById('stat-uptime').textContent = d.uptime_seconds;
    }
  } catch {}
}

// ── Init ───────────────────────────────────────────────────────────────────
fetchStatus();
setInterval(fetchStatus, 30000);
renderHistory();
updateDashboardStats();

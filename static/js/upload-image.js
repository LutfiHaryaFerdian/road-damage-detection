// ── Image Upload Detection ────────────────────────────────────────────────
let imgFile = null;

function imgDragOver(e) { e.preventDefault(); document.getElementById('img-drop-zone').classList.add('drag-over'); }
function imgDragLeave(e) { document.getElementById('img-drop-zone').classList.remove('drag-over'); }
function imgDrop(e) {
  e.preventDefault();
  imgDragLeave(e);
  const f = e.dataTransfer.files[0];
  if (f) imgLoadFile(f);
}
function imgFileSelected(e) { if (e.target.files[0]) imgLoadFile(e.target.files[0]); }

function imgLoadFile(f) {
  const allowed = ['image/jpeg','image/png','image/webp','image/jpg'];
  if (!allowed.includes(f.type)) { alert('Format tidak didukung. Gunakan JPG/PNG/WEBP'); return; }
  imgFile = f;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('img-preview').src = ev.target.result;
    document.getElementById('img-preview-wrap').classList.remove('hidden');
    document.getElementById('img-drop-zone').classList.add('hidden');
    const btn = document.getElementById('img-detect-btn');
    btn.disabled = false; btn.classList.remove('opacity-50','cursor-not-allowed');
    document.getElementById('img-detect-label').textContent = '🔍 Deteksi Kerusakan';
    document.getElementById('img-result-placeholder').classList.remove('hidden');
    document.getElementById('img-result').classList.add('hidden');
    document.getElementById('img-loading').classList.add('hidden');
  };
  reader.readAsDataURL(f);
}

function imgReset() {
  imgFile = null;
  document.getElementById('img-file-input').value = '';
  document.getElementById('img-preview-wrap').classList.add('hidden');
  document.getElementById('img-drop-zone').classList.remove('hidden');
  const btn = document.getElementById('img-detect-btn');
  btn.disabled = true; btn.classList.add('opacity-50','cursor-not-allowed');
  document.getElementById('img-detect-label').textContent = 'Pilih gambar terlebih dahulu';
  document.getElementById('img-result').classList.add('hidden');
  document.getElementById('img-result-placeholder').classList.remove('hidden');
}

async function imgDetect() {
  if (!imgFile) return;
  const btn = document.getElementById('img-detect-btn');
  btn.disabled = true;
  document.getElementById('img-loading').classList.remove('hidden');
  document.getElementById('img-loading').classList.add('flex');
  document.getElementById('img-result').classList.add('hidden');
  document.getElementById('img-result-placeholder').classList.add('hidden');
  document.getElementById('img-detect-label').textContent = 'Memproses...';

  const fd = new FormData();
  fd.append('file', imgFile);

  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || 'Gagal');
    imgShowResult(d);
  } catch (e) {
    alert('❌ Error: ' + e.message);
  } finally {
    document.getElementById('img-loading').classList.add('hidden');
    document.getElementById('img-loading').classList.remove('flex');
    btn.disabled = false;
    document.getElementById('img-detect-label').textContent = '🔍 Deteksi Ulang';
  }
}

function imgShowResult(d) {
  const ts = new Date().toLocaleTimeString('id-ID');
  document.getElementById('img-result-img').src = d.result_image_url + '?t=' + Date.now();
  document.getElementById('img-download-btn').href = d.result_image_url;
  document.getElementById('img-result-placeholder').classList.add('hidden');
  document.getElementById('img-result').classList.remove('hidden');

  // Stats row
  const overallColor = d.overall_color || '#3b82f6';
  document.getElementById('img-stats-row').innerHTML = `
    <div class="stat-mini">
      <div class="text-2xl font-bold" style="color:${overallColor}">${d.total}</div>
      <div class="text-xs text-gray-400">Total Deteksi</div>
    </div>
    <div class="stat-mini">
      <div class="text-lg font-bold" style="color:${overallColor}">${d.overall_icon} ${d.overall_severity}</div>
      <div class="text-xs text-gray-400">Tingkat Keparahan</div>
    </div>
    <div class="stat-mini">
      <div class="text-lg font-bold text-cyan-400">${d.total_damage_area_pct}%</div>
      <div class="text-xs text-gray-400">Area Rusak</div>
    </div>
  `;

  // Detection table
  const wrap = document.getElementById('img-det-table-wrap');
  if (d.detections && d.detections.length > 0) {
    const avgConf = d.detections.reduce((s,x)=>s+x.confidence,0) / d.detections.length;
    const rows = d.detections.map((det, i) => `
      <tr>
        <td class="text-gray-500">${i+1}</td>
        <td><span style="color:${det.color}">${det.emoji} ${det.label}</span></td>
        <td>
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs">${(det.confidence*100).toFixed(1)}%</span>
            <div class="conf-bar-bg flex-1" style="min-width:40px">
              <div class="conf-bar-fill" style="width:${(det.confidence*100).toFixed(0)}%;background:${det.color}"></div>
            </div>
          </div>
        </td>
        <td><span class="text-xs font-mono" style="color:${det.severity_color}">${det.severity_icon} ${det.severity}</span></td>
      </tr>
    `).join('');
    wrap.innerHTML = `
      <div class="mt-3 overflow-x-auto">
        <table class="det-table">
          <thead><tr><th>#</th><th>Jenis</th><th>Confidence</th><th>Severity</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    // Add to history
    addHistory({
      mode: 'image', total: d.total,
      summary: d.summary.map(s=>`${s.emoji}${s.count} ${s.label}`).join(', '),
      avg_conf: avgConf, time: ts,
    });
  } else {
    wrap.innerHTML = '<div class="text-center py-6 text-gray-500">✅ Tidak ada kerusakan terdeteksi</div>';
    addHistory({ mode: 'image', total: 0, summary: 'Tidak ada kerusakan', time: ts });
  }
}

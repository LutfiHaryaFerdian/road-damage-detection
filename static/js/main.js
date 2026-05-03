/* ===================================================
   Road Damage AI — Frontend Logic
   =================================================== */

// ── Element refs ─────────────────────────────────────────────────────────────
const dropZone          = document.getElementById("dropZone");
const fileInput         = document.getElementById("fileInput");
const dropIcon          = document.getElementById("dropIcon");
const dropTitle         = document.getElementById("dropTitle");
const dropSub           = document.getElementById("dropSub");
const previewWrap       = document.getElementById("previewWrap");
const previewImg        = document.getElementById("previewImg");
const resetBtn          = document.getElementById("resetBtn");
const detectBtn         = document.getElementById("detectBtn");
const detectLabel       = document.getElementById("detectLabel");

const uploadSection     = document.getElementById("uploadSection");
const loadingSection    = document.getElementById("loadingSection");
const resultSection     = document.getElementById("resultSection");
const errorSection      = document.getElementById("errorSection");

const statsRow          = document.getElementById("statsRow");
const resultImg         = document.getElementById("resultImg");
const downloadBtn       = document.getElementById("downloadBtn");
const detectionTableWrap= document.getElementById("detectionTableWrap");
const detectionTableBody= document.getElementById("detectionTableBody");
const noDetectMsg       = document.getElementById("noDetectMsg");

const newDetectBtn      = document.getElementById("newDetectBtn");
const errorMsg          = document.getElementById("errorMsg");
const errorRetryBtn     = document.getElementById("errorRetryBtn");

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── Drop zone interactions ────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

// ── File selection ────────────────────────────────────────────────────────────
function setFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  if (!allowed.includes(file.type)) {
    showError("Format tidak didukung. Gunakan JPG, PNG, atau WEBP.");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("Ukuran file melebihi 16 MB.");
    return;
  }

  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;

  dropZone.classList.add("hidden");
  previewWrap.classList.remove("hidden");

  detectBtn.disabled = false;
  detectLabel.textContent = `🚀 Deteksi Kerusakan — ${file.name}`;
}

resetBtn.addEventListener("click", resetUpload);
function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.classList.add("hidden");
  dropZone.classList.remove("hidden");
  detectBtn.disabled = true;
  detectLabel.textContent = "Pilih gambar terlebih dahulu";
  showSection("upload");
}

// ── Detection ─────────────────────────────────────────────────────────────────
detectBtn.addEventListener("click", runDetection);

async function runDetection() {
  if (!selectedFile) return;

  showSection("loading");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || `Server error ${response.status}`);
      return;
    }

    renderResults(data);
    showSection("result");

  } catch (err) {
    showError("Gagal menghubungi server. Periksa koneksi Anda.");
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  // Result image
  resultImg.src = data.result_image_url + "?t=" + Date.now();
  downloadBtn.href = data.result_image_url;

  // Stats chips
  statsRow.innerHTML = "";

  const totalChip = makeChip("🎯", `${data.total} Kerusakan`, "#3b82f6");
  statsRow.appendChild(totalChip);

  (data.summary || []).forEach(s => {
    statsRow.appendChild(makeChip(s.emoji, `${s.label}`, s.color, s.count));
  });

  // Table / no detect
  if (data.detections && data.detections.length > 0) {
    detectionTableBody.innerHTML = "";
    data.detections.forEach((det, i) => {
      const tr = document.createElement("tr");
      const [x1, y1, x2, y2] = det.bbox;
      const confPct = Math.round(det.confidence * 100);
      const barWidth = Math.max(4, confPct);

      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>
          <span class="damage-badge" style="--badge-color:${det.color}">
            ${det.emoji} ${det.label}
          </span>
        </td>
        <td>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:${barWidth}%; --bar-color:${det.color}"></div>
            <span class="conf-pct">${confPct}%</span>
          </div>
        </td>
        <td style="font-size:0.78rem;font-family:monospace">
          (${x1}, ${y1}) → (${x2}, ${y2})
        </td>
      `;
      detectionTableBody.appendChild(tr);
    });

    detectionTableWrap.classList.remove("hidden");
    noDetectMsg.classList.add("hidden");
  } else {
    detectionTableWrap.classList.add("hidden");
    noDetectMsg.classList.remove("hidden");
  }
}

function makeChip(emoji, label, color, count = null) {
  const div = document.createElement("div");
  div.className = "stat-chip";
  div.style.setProperty("--chip-color", color);
  div.innerHTML = `${emoji} ${label}${count !== null ? ` <span class="chip-count">${count}</span>` : ""}`;
  return div;
}

// ── Section visibility ────────────────────────────────────────────────────────
function showSection(name) {
  uploadSection.classList.add("hidden");
  loadingSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");

  if (name === "upload")  uploadSection.classList.remove("hidden");
  if (name === "loading") loadingSection.classList.remove("hidden");
  if (name === "result")  resultSection.classList.remove("hidden");
  if (name === "error")   errorSection.classList.remove("hidden");
}

function showError(msg) {
  errorMsg.textContent = msg;
  showSection("error");
}

// ── Action buttons ────────────────────────────────────────────────────────────
newDetectBtn.addEventListener("click", resetUpload);
errorRetryBtn.addEventListener("click", () => {
  showSection("upload");
});

// ── Particle animation ────────────────────────────────────────────────────────
(function spawnParticles() {
  const container = document.getElementById("particles");
  if (!container) return;

  function createDot() {
    const dot = document.createElement("div");
    const size = Math.random() * 3 + 1;
    const x    = Math.random() * 100;
    const dur  = Math.random() * 20 + 15;
    const delay= Math.random() * 10;

    dot.style.cssText = `
      position:absolute;
      width:${size}px; height:${size}px;
      border-radius:50%;
      background:rgba(59,130,246,${Math.random() * 0.4 + 0.1});
      left:${x}%;
      bottom:-10px;
      animation: floatUp ${dur}s ${delay}s linear infinite;
      pointer-events:none;
    `;
    container.appendChild(dot);
  }

  // Inject keyframes
  const style = document.createElement("style");
  style.textContent = `
    @keyframes floatUp {
      0%   { transform: translateY(0) scale(1);  opacity: 0; }
      10%  { opacity: 1; }
      90%  { opacity: 0.5; }
      100% { transform: translateY(-100vh) scale(0.5); opacity: 0; }
    }
  `;
  document.head.appendChild(style);

  for (let i = 0; i < 28; i++) createDot();
})();

const TREATMENTS = [
  { id: "auto", name: "Auto", note: "Full bath" },
  { id: "vintage", name: "Vintage", note: "Fade & scratches" },
  { id: "denoise", name: "Denoise", note: "Grain & push" },
  { id: "color", name: "Color", note: "Cast & dye" },
  { id: "sharpen", name: "Sharpen", note: "Lift the blur" },
  { id: "upscale", name: "Resolve", note: "2× FSRCNN" },
  { id: "portrait", name: "Portrait", note: "Face-aware" },
  { id: "document", name: "Document", note: "Paper & ink" },
];

const BATH = [
  "Registering the trace",
  "Lifting the dye",
  "Filling the tears",
  "Keeping the drawing",
  "Fixing the silver",
];

const state = {
  file: null,
  name: "photograph.jpg",
  beforeUrl: null,
  afterUrl: null,
  traceFile: null,
  traceUrl: null,
  mode: "auto",
  split: 0.5,
  busy: false,
  meta: null,
};

const $ = (id) => document.getElementById(id);

const els = {
  file: $("file"),
  openBtn: $("openBtn"),
  restoreBtn: $("restoreBtn"),
  downloadBtn: $("downloadBtn"),
  saveChip: $("saveChip"),
  empty: $("empty"),
  frame: $("frame"),
  stage: $("stage"),
  compare: $("compare"),
  beforeClip: $("beforeClip"),
  beforeImg: $("beforeImg"),
  afterImg: $("afterImg"),
  divider: $("divider"),
  treatments: $("treatments"),
  strength: $("strength"),
  strengthVal: $("strengthVal"),
  upscale: $("upscale"),
  meta: $("meta"),
  strip: $("strip"),
  developing: $("developing"),
  bathLabel: $("bathLabel"),
  traceFile: $("traceFile"),
  traceBtn: $("traceBtn"),
  traceClear: $("traceClear"),
  traceNote: $("traceNote"),
  traceThumb: $("traceThumb"),
};

function setSplit(t) {
  state.split = Math.min(0.97, Math.max(0.03, t));
  const pct = `${state.split * 100}%`;
  els.beforeClip.style.clipPath = `inset(0 ${(1 - state.split) * 100}% 0 0)`;
  els.divider.style.left = pct;
}

function showPlate() {
  els.empty.classList.add("hidden");
  els.empty.style.display = "none";
  els.frame.classList.remove("hidden");
}

function setTrace(file, url, label) {
  if (state.traceUrl) URL.revokeObjectURL(state.traceUrl);
  state.traceFile = file || null;
  state.traceUrl = url || null;
  if (els.traceThumb) {
    if (url) {
      els.traceThumb.src = url;
      els.traceThumb.classList.remove("hidden");
    } else {
      els.traceThumb.removeAttribute("src");
      els.traceThumb.classList.add("hidden");
    }
  }
  if (els.traceClear) els.traceClear.classList.toggle("hidden", !file);
  if (els.traceNote) {
    els.traceNote.textContent = file
      ? `Tracing from ${label || file.name}`
      : "No guide — repair from the worn print alone.";
  }
}

function setBefore(url, name) {
  if (state.beforeUrl) URL.revokeObjectURL(state.beforeUrl);
  if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
  state.beforeUrl = url;
  state.afterUrl = null;
  state.name = name || "photograph.jpg";
  state.meta = null;
  els.beforeImg.src = url;
  els.afterImg.src = url;
  showPlate();
  setSplit(0.5);
  els.restoreBtn.disabled = false;
  els.downloadBtn.classList.add("disabled");
  els.downloadBtn.removeAttribute("href");
  if (els.saveChip) els.saveChip.classList.add("hidden");
  els.meta.textContent = state.traceFile
    ? `${state.name} · trace loaded · waiting`
    : `${state.name} · waiting for the bath`;
}

function downloadName() {
  const base = (state.name || "photograph").replace(/\.[^.]+$/, "");
  return `${base}-restored.jpg`;
}

function setAfter(blob, headers) {
  if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
  state.afterUrl = URL.createObjectURL(blob);
  els.afterImg.src = state.afterUrl;
  setSplit(0.48);
  const filename = headers.get("X-Gefpan-Filename") || downloadName();
  els.downloadBtn.classList.remove("disabled");
  els.downloadBtn.href = state.afterUrl;
  els.downloadBtn.setAttribute("download", filename);
  if (els.saveChip) {
    els.saveChip.classList.remove("hidden");
    els.saveChip.href = state.afterUrl;
    els.saveChip.setAttribute("download", filename);
  }
  const w = headers.get("X-Gefpan-Width");
  const h = headers.get("X-Gefpan-Height");
  const ms = headers.get("X-Gefpan-Ms");
  const engine = headers.get("X-Gefpan-Engine");
  const mode = headers.get("X-Gefpan-Mode");
  const traced = headers.get("X-Gefpan-Trace") === "1";
  state.meta = { w, h, ms, engine, mode, traced };
  els.meta.textContent = traced
    ? `${w}×${h} · traced repair · ${engine} · ${ms} ms · ready to download`
    : `${w}×${h} · ${mode} · ${engine} · ${ms} ms · ready to download`;
}

function renderTreatments() {
  els.treatments.innerHTML = "";
  for (const t of TREATMENTS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (t.id === state.mode ? " active" : "");
    btn.innerHTML = `${t.name}<small>${t.note}</small>`;
    btn.addEventListener("click", () => {
      state.mode = t.id;
      renderTreatments();
      if (t.id === "upscale") els.upscale.checked = true;
    });
    els.treatments.appendChild(btn);
  }
}

function takeFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.file = file;
  setBefore(URL.createObjectURL(file), file.name);
}

function takeTrace(file, label) {
  if (!file || !file.type.startsWith("image/")) return;
  setTrace(file, URL.createObjectURL(file), label);
}

async function restore() {
  if (!state.file || state.busy) return;
  state.busy = true;
  els.restoreBtn.disabled = true;
  els.developing.hidden = false;
  let i = 0;
  els.bathLabel.textContent = state.traceFile ? BATH[0] : "Bathing the plate";
  const tick = setInterval(() => {
    i = (i + 1) % BATH.length;
    els.bathLabel.textContent = BATH[i];
  }, 900);

  const body = new FormData();
  body.append("file", state.file, state.name);
  body.append("mode", state.mode);
  body.append("strength", String(Number(els.strength.value) / 100));
  body.append("upscale", els.upscale.checked ? "true" : "false");
  if (state.traceFile) {
    body.append("trace", state.traceFile, state.traceFile.name || "trace.jpg");
  }

  try {
    const res = await fetch("/api/restore", { method: "POST", body });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    const blob = await res.blob();
    setAfter(blob, res.headers);
  } catch (err) {
    els.meta.textContent = `The bath failed — ${err.message}`;
  } finally {
    clearInterval(tick);
    els.developing.hidden = true;
    state.busy = false;
    els.restoreBtn.disabled = !state.file;
  }
}

function download(ev) {
  if (ev) ev.preventDefault();
  if (!state.afterUrl) return;
  const a = document.createElement("a");
  a.href = state.afterUrl;
  a.download = downloadName();
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function bindCompare() {
  let dragging = false;
  const pos = (ev) => {
    const r = els.compare.getBoundingClientRect();
    const x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    setSplit(x / r.width);
  };
  els.compare.addEventListener("pointerdown", (ev) => {
    dragging = true;
    els.compare.setPointerCapture(ev.pointerId);
    pos(ev);
  });
  els.compare.addEventListener("pointermove", (ev) => {
    if (dragging) pos(ev);
  });
  els.compare.addEventListener("pointerup", () => {
    dragging = false;
  });
}

function bindDrop() {
  const over = (on) => els.stage.classList.toggle("drag", on);
  ["dragenter", "dragover"].forEach((ev) =>
    els.stage.addEventListener(ev, (e) => {
      e.preventDefault();
      over(true);
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    els.stage.addEventListener(ev, (e) => {
      e.preventDefault();
      over(false);
    })
  );
  els.stage.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    takeFile(file);
  });
}

async function loadSamples() {
  let items = [];
  try {
    const res = await fetch("/api/samples");
    items = await res.json();
  } catch {
    items = [];
  }
  els.strip.innerHTML = "";
  for (const item of items) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "card";
    btn.innerHTML = `
      <img src="${item.file}" alt="${item.title}" />
      <div class="cap"><b>${item.title}</b><span>${item.year}</span></div>
    `;
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".card").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const wornRes = await fetch(item.file);
      const wornBlob = await wornRes.blob();
      const worn = new File([wornBlob], `${item.id}.jpg`, { type: "image/jpeg" });
      state.mode = item.mode || "auto";
      renderTreatments();
      takeFile(worn);
      if (item.trace) {
        try {
          const tr = await fetch(item.trace);
          const tb = await tr.blob();
          takeTrace(new File([tb], `${item.id}-trace.jpg`, { type: "image/jpeg" }), `${item.title} guide`);
        } catch {
          setTrace(null, null);
        }
      } else {
        setTrace(null, null);
      }
      els.meta.textContent = `${item.title} · tracing from the clean plate`;
    });
    els.strip.appendChild(btn);
  }
}

els.openBtn.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => takeFile(els.file.files[0]));
if (els.traceBtn) els.traceBtn.addEventListener("click", () => els.traceFile.click());
if (els.traceFile) {
  els.traceFile.addEventListener("change", () => takeTrace(els.traceFile.files[0]));
}
if (els.traceClear) {
  els.traceClear.addEventListener("click", () => setTrace(null, null));
}
els.restoreBtn.addEventListener("click", restore);
els.downloadBtn.addEventListener("click", download);
if (els.saveChip) els.saveChip.addEventListener("click", download);
els.strength.addEventListener("input", () => {
  els.strengthVal.textContent = els.strength.value;
});

window.addEventListener("keydown", (ev) => {
  if (ev.target.matches("input, textarea")) return;
  if (ev.code === "Space" && state.afterUrl) {
    ev.preventDefault();
    els.afterImg.style.opacity = "0";
  }
  if (ev.key === "Enter") restore();
  if (ev.key.toLowerCase() === "d") download();
});
window.addEventListener("keyup", (ev) => {
  if (ev.code === "Space") els.afterImg.style.opacity = "1";
});

renderTreatments();
bindCompare();
bindDrop();
loadSamples();

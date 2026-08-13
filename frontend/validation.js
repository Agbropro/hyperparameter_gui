const state = { jobs: [], selectedId: null, timer: null };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const colors = ["#795cff", "#8fb71d", "#ff714b", "#278bb8", "#d84d96", "#8a6842", "#15a37d", "#a758d1"];
const metricOrder = ["precision", "recall", "f1", "map50", "map75", "map50_95", "mask_precision", "mask_recall", "mask_f1", "mask_map50", "mask_map75", "mask_map50_95", "accuracy_top1", "accuracy_top5", "fitness", "inference_ms"];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function escapeHtml(value) { const node = document.createElement("div"); node.textContent = value ?? ""; return node.innerHTML; }
function metricLabel(name) { return name.replaceAll("_", " ").replace("map50 95", "mAP50–95").replace("map50", "mAP50").replace("map75", "mAP75"); }
function metricValue(value, name) { return name === "inference_ms" ? Number(value).toFixed(2) : Number(value).toFixed(4); }

function renderModelInputs() {
  const root = $("#model-inputs");
  const existing = $$(".model-row", root).map(row => ({ label: $("[data-label]", row).value, path: $("[data-path]", row).value }));
  const count = Math.max(1, Math.min(20, Number($("#model-count").value) || 1));
  $("#model-count").value = count;
  root.innerHTML = Array.from({ length: count }, (_, index) => `
    <div class="model-row"><span class="model-index">${String(index + 1).padStart(2, "0")}</span>
      <input data-label placeholder="Model label" value="${escapeHtml(existing[index]?.label || `Model ${index + 1}`)}" required />
      <input data-path placeholder="/path/to/weights/best.pt" value="${escapeHtml(existing[index]?.path || "")}" required />
    </div>`).join("");
}

async function submitValidation(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const message = $("#validation-message");
  const button = $(".primary", event.target);
  message.hidden = true; button.disabled = true;
  const models = $$(".model-row").map(row => ({ label: $("[data-label]", row).value, path: $("[data-path]", row).value }));
  const body = {
    name: form.get("name"), dataset: form.get("dataset"), models,
    confidence: Number(form.get("confidence")), iou: Number(form.get("iou")),
    image_size: Number(form.get("image_size")), batch: Number(form.get("batch")), device: form.get("device") || null
  };
  try {
    const job = await api("/api/validation/jobs", { method: "POST", body: JSON.stringify(body) });
    state.selectedId = job.id;
    await loadJobs();
  } catch (error) { message.textContent = error.message; message.hidden = false; }
  finally { button.disabled = false; }
}

async function loadJobs() {
  try { state.jobs = await api("/api/validation/jobs"); renderHistory(); renderSelected(); }
  catch (error) { $("#validation-jobs").innerHTML = `<div class="message">${escapeHtml(error.message)}</div>`; }
  const active = state.jobs.some(job => ["queued", "running"].includes(job.status));
  clearTimeout(state.timer); if (active) state.timer = setTimeout(loadJobs, 2500);
}

function renderHistory() {
  const root = $("#validation-jobs");
  if (!state.jobs.length) { root.innerHTML = '<div class="empty"><span>◇</span><p>No comparisons yet</p><small>Validation jobs will appear here.</small></div>'; return; }
  if (!state.selectedId) state.selectedId = state.jobs[0].id;
  root.innerHTML = state.jobs.map(job => {
    const finished = job.models.filter(model => ["completed", "failed"].includes(model.status)).length;
    const percent = Math.round(finished / job.models.length * 100);
    const hasFailure = job.models.some(model => model.status === "failed");
    return `<article class="validation-job ${job.id === state.selectedId ? "selected" : ""}" data-job="${job.id}">
      <div class="validation-job-head"><b>${escapeHtml(job.name)}</b><span class="badge ${job.status}">${job.status}</span></div>
      <div class="validation-job-meta">${finished}/${job.models.length} MODELS · CONF ${job.confidence} · IOU ${job.iou}</div>
      <div class="mini-progress"><i style="width:${percent}%"></i></div>
      ${hasFailure ? `<button class="retry-validation" data-retry="${job.id}">Retry failed models</button>` : ""}
    </article>`;
  }).join("");
  $$('[data-job]').forEach(card => card.addEventListener("click", () => { state.selectedId = card.dataset.job; renderHistory(); renderSelected(); }));
  $$('[data-retry]').forEach(button => button.addEventListener("click", async event => { event.stopPropagation(); await retryJob(button.dataset.retry); }));
}

async function retryJob(id) {
  try { await api(`/api/validation/jobs/${id}/retry`, { method: "POST" }); state.selectedId = id; await loadJobs(); }
  catch (error) { window.alert(error.message); }
}

function selectedJob() { return state.jobs.find(job => job.id === state.selectedId); }
function availableMetrics(job) {
  const names = new Set(job.models.flatMap(model => Object.keys(model.metrics || {})));
  return metricOrder.filter(name => names.has(name)).concat([...names].filter(name => !metricOrder.includes(name)));
}

function renderSelected() {
  const job = selectedJob(); const root = $("#comparison-results");
  if (!job) { root.hidden = true; return; }
  root.hidden = false;
  $("#comparison-name").textContent = job.name;
  $("#comparison-settings").textContent = `TEST SPLIT · CONF ${job.confidence} · IOU ${job.iou} · IMGSZ ${job.image_size} · BATCH ${job.batch}`;
  const metrics = availableMetrics(job); const select = $("#chart-metric"); const previous = select.value;
  select.innerHTML = metrics.map(name => `<option value="${name}">${metricLabel(name)}</option>`).join("");
  select.value = metrics.includes(previous) ? previous : (metrics.includes("map50_95") ? "map50_95" : metrics[0] || "");
  renderTable(job, metrics); renderPerClass(job); drawChart();
}

function renderTable(job, metrics) {
  $("#comparison-head").innerHTML = `<tr><th>Model</th><th>Status</th>${metrics.map(name => `<th>${metricLabel(name)}</th>`).join("")}<th>Time</th></tr>`;
  const best = Object.fromEntries(metrics.map(name => {
    const values = job.models.filter(model => model.status === "completed" && model.metrics[name] != null).map(model => Number(model.metrics[name]));
    return [name, values.length ? (name === "inference_ms" ? Math.min(...values) : Math.max(...values)) : null];
  }));
  $("#comparison-body").innerHTML = job.models.map(model => `<tr><td><b>${escapeHtml(model.label)}</b><br><span title="${escapeHtml(model.model_path)}">${escapeHtml(model.model_path.split("/").pop())}</span></td><td>${model.status}${model.error ? `<div class="model-failure">${escapeHtml(model.error)}</div>` : ""}</td>${metrics.map(name => {
    const value = model.metrics[name]; const winner = value != null && Number(value) === best[name];
    return `<td class="${winner ? "best-cell" : ""}">${value == null ? "—" : metricValue(value, name)}</td>`;
  }).join("")}<td>${model.duration_seconds == null ? "—" : `${model.duration_seconds}s`}</td></tr>`).join("");

  const metric = $("#chart-metric").value;
  const candidates = job.models.filter(model => model.metrics[metric] != null);
  if (!candidates.length) { $("#winner-callout").innerHTML = ""; return; }
  const winner = candidates.reduce((bestModel, model) => {
    const better = metric === "inference_ms" ? model.metrics[metric] < bestModel.metrics[metric] : model.metrics[metric] > bestModel.metrics[metric];
    return better ? model : bestModel;
  });
  $("#winner-callout").innerHTML = `<div class="winner-banner"><b>Best ${metricLabel(metric)} · ${escapeHtml(winner.label)}</b><span>${metricValue(winner.metrics[metric], metric)}</span></div>`;
}

function renderPerClass(job) {
  $("#per-class-results").innerHTML = job.models.filter(model => model.per_class?.length).map(model => {
    const keys = [...new Set(model.per_class.flatMap(row => Object.keys(row)))];
    return `<details class="class-details"><summary>${escapeHtml(model.label)} · per-class metrics (${model.per_class.length})</summary><div class="class-table-wrap"><table class="class-table"><thead><tr>${keys.map(key => `<th>${escapeHtml(key)}</th>`).join("")}</tr></thead><tbody>${model.per_class.map(row => `<tr>${keys.map(key => `<td>${escapeHtml(typeof row[key] === "number" ? Number(row[key]).toFixed(5) : row[key] ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`;
  }).join("");
}

function drawChart() {
  const canvas = $("#metric-chart"); const job = selectedJob(); const metric = $("#chart-metric").value;
  if (!job || !metric) return;
  const models = job.models.filter(model => model.metrics[metric] != null);
  const rect = canvas.getBoundingClientRect(); const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * ratio); canvas.height = Math.max(1, rect.height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
  if (!models.length) { ctx.fillStyle = "#727369"; ctx.font = "12px Manrope"; ctx.fillText("Metrics will appear after validation completes.", 20, 35); return; }
  const pad = { left: 55, right: 20, top: 24, bottom: 60 }; const width = rect.width - pad.left - pad.right; const height = rect.height - pad.top - pad.bottom;
  const maximum = Math.max(...models.map(model => Number(model.metrics[metric])), metric === "inference_ms" ? 1 : 1);
  ctx.strokeStyle = "rgba(18,19,15,.12)"; ctx.fillStyle = "#727369"; ctx.font = "9px DM Mono"; ctx.textAlign = "right";
  for (let tick = 0; tick <= 5; tick++) { const y = pad.top + height - height * tick / 5; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(rect.width - pad.right, y); ctx.stroke(); ctx.fillText((maximum * tick / 5).toFixed(metric === "inference_ms" ? 1 : 2), pad.left - 8, y + 3); }
  const slot = width / models.length; const barWidth = Math.min(80, slot * .58);
  models.forEach((model, index) => { const value = Number(model.metrics[metric]); const barHeight = value / maximum * height; const x = pad.left + slot * index + (slot - barWidth) / 2; const y = pad.top + height - barHeight; ctx.fillStyle = colors[index % colors.length]; ctx.beginPath(); ctx.roundRect(x, y, barWidth, barHeight, 5); ctx.fill(); ctx.fillStyle = "#12130f"; ctx.textAlign = "center"; ctx.font = "10px DM Mono"; ctx.fillText(metricValue(value, metric), x + barWidth / 2, Math.max(12, y - 7)); ctx.fillStyle = "#727369"; ctx.font = "9px Manrope"; const label = model.label.length > 18 ? `${model.label.slice(0, 16)}…` : model.label; ctx.fillText(label, x + barWidth / 2, pad.top + height + 22); });
}

function setupThreshold(name, digits) {
  let number = $(`#${name}`);
  let slider = $(`#${name}-slider`);

  // Upgrade the previous slider/output markup if an old HTML document was cached.
  if (!slider && number?.type === "range") {
    slider = number;
    slider.id = `${name}-slider`;
    slider.classList.add("threshold-slider");
    slider.removeAttribute("name");
    const output = name === "confidence" ? $("#conf-output") : $("#iou-output");
    number = document.createElement("input");
    number.type = "number";
    number.id = name;
    number.name = name;
    number.min = "0";
    number.max = "1";
    number.step = slider.step;
    number.value = slider.value;
    number.required = true;
    number.className = "threshold-number";
    if (output) output.replaceWith(number);
    else slider.parentElement.querySelector("div")?.append(number);
  }
  if (!number || !slider) return;
  const fromSlider = () => { number.value = Number(slider.value).toFixed(digits); };
  const fromNumber = () => {
    if (number.value === "") return;
    const value = Math.max(0, Math.min(1, Number(number.value)));
    number.value = Number.isFinite(value) ? value.toFixed(digits) : (0).toFixed(digits);
    slider.value = number.value;
  };
  slider.addEventListener("input", fromSlider);
  number.addEventListener("input", () => { if (number.validity.valid) slider.value = number.value; });
  number.addEventListener("change", fromNumber);
  fromNumber();
}

async function init() {
  renderModelInputs();
  setupThreshold("confidence", 3);
  setupThreshold("iou", 2);
  $("#model-count").addEventListener("change", renderModelInputs);
  $("#validation-form").addEventListener("submit", submitValidation);
  $("#refresh-validation").addEventListener("click", loadJobs);
  $("#chart-metric").addEventListener("change", () => { const job = selectedJob(); renderTable(job, availableMetrics(job)); drawChart(); });
  window.addEventListener("resize", drawChart);
  await loadJobs();
}
init();

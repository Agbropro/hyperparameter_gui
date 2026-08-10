const state = { options: null, experiments: [], timer: null };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const parameterHelp = {
  lr0: ["Initial learning rate", "Controls how large the first weight updates are. Higher learns faster but can become unstable; lower is steadier but may learn too slowly.", "Core"],
  lrf: ["Final learning-rate fraction", "The ending learning rate is lr0 × lrf. A smaller value slows learning more near the end, helping the model make finer final adjustments.", "Core"],
  momentum: ["Gradient momentum", "Carries part of previous updates into the next update. Higher values smooth noisy gradients, but too much can overshoot a good solution.", "Core"],
  weight_decay: ["Weight regularization", "Penalizes very large model weights to reduce overfitting. Too much can cause underfitting and reduce the model's ability to learn detail.", "Core"],
  warmup_epochs: ["Learning-rate warmup", "Number of early epochs that gradually ramp up the learning rate. More warmup can stabilize difficult starts but leaves fewer epochs at the full schedule.", "Core"],
  warmup_momentum: ["Starting warmup momentum", "Momentum used at the beginning of warmup, then gradually changed to the main momentum value. It influences stability during the first updates.", "Core"],
  box: ["Bounding-box loss weight", "Controls how strongly incorrect box location and size are penalized. Higher prioritizes box localization relative to other training objectives.", "Detect / Segment"],
  cls: ["Classification loss weight", "Controls how strongly wrong class predictions are penalized. Higher emphasizes choosing the correct class, potentially at the expense of localization.", "Detect / Segment"],
  dfl: ["Distribution focal loss weight", "Controls a box-localization loss that helps refine object boundaries. Higher puts more emphasis on precise bounding-box edges.", "Detect / Segment"],
  hsv_h: ["Hue variation", "Randomly shifts image colors around the color wheel. It helps when object identity should not depend on exact hue; too much can create unrealistic colors.", "Augmentation"],
  hsv_s: ["Saturation variation", "Randomly changes color intensity. It can improve robustness to cameras and lighting, while high values may remove useful color information.", "Augmentation"],
  hsv_v: ["Brightness variation", "Randomly makes training images brighter or darker. It improves lighting robustness, but extreme values may hide important image detail.", "Augmentation"],
  degrees: ["Rotation range", "Randomly rotates images by up to this many degrees in either direction. Useful for rotated viewpoints; keep it small when objects always stay upright.", "Augmentation"],
  translate: ["Translation range", "Randomly shifts images horizontally and vertically by a fraction of image size. It teaches position tolerance, but large shifts can crop objects heavily.", "Augmentation"],
  scale: ["Scale variation", "Randomly zooms images in or out. It teaches the model to recognize different object sizes; extreme scaling can lose objects or context.", "Augmentation"],
  shear: ["Shear range", "Randomly slants images by up to this many degrees. It simulates viewpoint distortion, but strong shear may make images unrealistic.", "Augmentation"],
  perspective: ["Perspective distortion", "Randomly changes perspective to simulate different viewing angles. This is very sensitive, which is why useful values are usually small.", "Augmentation"],
  flipud: ["Vertical-flip probability", "Chance from 0 to 1 that an image is flipped upside down. Use near zero when upside-down examples are impossible in real use.", "Augmentation"],
  fliplr: ["Horizontal-flip probability", "Chance from 0 to 1 that an image is mirrored left-to-right. Often useful unless left/right direction changes the label or meaning.", "Augmentation"],
  mosaic: ["Mosaic probability", "Chance of combining four images into one training image. It adds object-scale and context variety, but can create scenes unlike your real data.", "Detect / Segment"],
  mixup: ["MixUp probability", "Chance of blending two images and their labels. It can reduce overfitting, though large values can make training images look unnatural.", "Augmentation"],
  copy_paste: ["Copy-paste probability", "Chance of copying segmented objects between images. It can create more object arrangements and is mainly useful for segmentation data.", "Segment"],
  batch: ["Images per update", "Randomly chooses how many images are processed before each weight update. Larger batches use more GPU memory and give smoother gradients; smaller batches use less memory and update more noisily.", "Search choice"],
  optimizer: ["Weight-update algorithm", "Randomly chooses the algorithm that turns gradients into weight updates. SGD is often steady, Adam adapts quickly, and AdamW combines adaptive updates with decoupled weight decay.", "Search choice"],
  seed: ["Reproducibility seed", "Controls both the random hyperparameter sequence and YOLO training randomness. This app keeps the same seed across trials for a fairer comparison. Exact repeatability can still depend on hardware and libraries.", "Fixed value"]
};

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function currentTask() { return $('[name="task"]:checked').value; }
function pretty(value) { return String(value).replaceAll("_", " "); }

function updateTaskOptions() {
  const task = currentTask();
  $("#metrics").innerHTML = state.options.metrics[task].map((metric, index) =>
    `<label><input type="checkbox" name="metrics" value="${metric}" ${index === 0 ? "checked" : ""}><span>${pretty(metric)}</span></label>`
  ).join("");
  updateResolvedModel();
}

function updateResolvedModel() {
  const version = $("#yolo-version").value;
  const size = $("#model-size").value;
  const suffix = { detect: "", segment: "-seg", classify: "-cls" }[currentTask()];
  $("#resolved-model").textContent = `${version}${size}${suffix}.pt`;
  $("#version-note").textContent = state.options.versions[version]?.note || "";
}

function renderSearchSpace() {
  const ranges = state.options.search_space.ranges;
  $("#search-space").innerHTML = Object.entries(ranges).map(([name, range]) => `
    <label class="range-row" title="${name}"><span>${name}</span>
      <input data-range="${name}" data-edge="low" type="number" step="any" value="${range.low}" aria-label="${name} minimum">
      <input data-range="${name}" data-edge="high" type="number" step="any" value="${range.high}" aria-label="${name} maximum">
    </label>`).join("");
}

function renderParameterHelp() {
  const rangeLabels = {};
  $$('[data-range]').forEach(input => {
    const name = input.dataset.range;
    rangeLabels[name] ||= {};
    rangeLabels[name][input.dataset.edge] = input.value;
  });
  $("#parameter-help").innerHTML = Object.entries(parameterHelp).map(([name, [title, description, tag]]) => `
    <article class="help-card ${["batch", "optimizer", "seed"].includes(name) ? "help-special" : ""}">
      <div class="help-card-head"><h3>${name}</h3><span class="help-tag">${tag}</span></div>
      ${["batch", "optimizer", "seed"].includes(name) ? "" : `<p class="help-value-range"><span>Random range</span>${rangeLabels[name]?.low ?? "—"} → ${rangeLabels[name]?.high ?? "—"}</p>`}
      <p><strong>${title}.</strong> ${description}</p>
    </article>`).join("");
}

function switchRangeView(view) {
  const showingHelp = view === "help";
  if (showingHelp) renderParameterHelp();
  $("#range-config-view").hidden = showingHelp;
  $("#range-help-view").hidden = !showingHelp;
  $$('[data-range-view]').forEach(button => button.classList.toggle("active", button.dataset.rangeView === view));
}

async function inspectDataset() {
  const status = $("#dataset-status");
  status.className = "dataset-status"; status.textContent = "Inspecting…";
  try {
    const data = await api("/api/datasets/inspect", { method: "POST", body: JSON.stringify({ path: $("#dataset").value, task: currentTask() }) });
    $("#dataset").value = data.dataset;
    const found = Object.keys(data.folders).join(", ") || "paths declared in YAML";
    status.classList.add("valid"); status.textContent = `Ready · found ${found}`;
  } catch (error) { status.classList.add("invalid"); status.textContent = error.message; }
}

function collectRanges() {
  const defaults = state.options.search_space.ranges; const ranges = {};
  $$('[data-range]').forEach(input => {
    const name = input.dataset.range;
    ranges[name] ||= { ...defaults[name] };
    ranges[name][input.dataset.edge] = Number(input.value);
  });
  return ranges;
}

async function submitExperiment(event) {
  event.preventDefault();
  const button = $('.primary', event.target); const message = $("#form-message");
  button.disabled = true; message.hidden = true;
  const form = new FormData(event.target);
  const body = {
    name: form.get("name"), task: currentTask(), model: $("#resolved-model").textContent, dataset: form.get("dataset"),
    device: form.get("device") || null, trials: Number(form.get("trials")), epochs: Number(form.get("epochs")),
    image_size: Number(form.get("image_size")), seed: Number($("#random-seed").value), metrics: form.getAll("metrics"), ranges: collectRanges(),
    choices: {
      batch: $("#batch-choices").value.split(",").map(Number).filter(Number.isFinite),
      optimizer: $("#optimizer-choices").value.split(",").map(v => v.trim()).filter(Boolean)
    }
  };
  try {
    if (!body.metrics.length) throw new Error("Select at least one metric.");
    await api("/api/experiments", { method: "POST", body: JSON.stringify(body) });
    await loadExperiments();
  } catch (error) { message.textContent = error.message; message.hidden = false; }
  finally { button.disabled = false; }
}

function bestTrial(experiment) { return experiment.trials.find(t => t.number === experiment.best_trial); }
function renderExperiments() {
  const root = $("#experiments");
  if (!state.experiments.length) { root.innerHTML = '<div class="empty"><span>◇</span><p>No experiments yet</p><small>Your training runs will appear here.</small></div>'; return; }
  root.innerHTML = state.experiments.map(exp => {
    const done = exp.trials.filter(t => ["completed", "failed"].includes(t.status)).length;
    const percent = Math.round(done / exp.config.trials * 100); const best = bestTrial(exp);
    return `<article class="experiment" data-id="${exp.id}">
      <div class="exp-top"><b>${escapeHtml(exp.config.name)}</b><span class="badge ${exp.status}">${exp.status}</span></div>
      <div class="exp-meta">${exp.config.task.toUpperCase()} · ${done}/${exp.config.trials} TRIALS · ${exp.config.epochs} EPOCHS</div>
      <div class="progress"><i style="width:${percent}%"></i></div>
      <div class="score"><span>Best objective</span><b>${best?.score?.toFixed(4) ?? "—"}</b></div>
    </article>`;
  }).join("");
  $$(".experiment", root).forEach(card => card.addEventListener("click", () => showDetail(card.dataset.id)));
}

async function loadExperiments() {
  try { state.experiments = await api("/api/experiments"); renderExperiments(); }
  catch (error) { $("#experiments").innerHTML = `<div class="message">${escapeHtml(error.message)}</div>`; }
  const active = state.experiments.some(e => ["queued", "running"].includes(e.status));
  clearTimeout(state.timer); if (active) state.timer = setTimeout(loadExperiments, 2500);
}

function showDetail(id) {
  const exp = state.experiments.find(item => item.id === id); if (!exp) return;
  const best = bestTrial(exp); const metricNames = [...new Set(exp.trials.flatMap(t => Object.keys(t.metrics)))];
  $("#experiment-detail").innerHTML = `<div class="detail-title"><span class="badge ${exp.status}">${exp.status}</span><h2>${escapeHtml(exp.config.name)}</h2><p>${exp.config.model} · ${escapeHtml(exp.config.dataset)}</p></div>
    <div class="stats"><div class="stat"><span>Best score</span><b>${best?.score?.toFixed(4) ?? "—"}</b></div><div class="stat"><span>Trials complete</span><b>${exp.trials.length}/${exp.config.trials}</b></div><div class="stat"><span>Objective</span><b>${exp.config.metrics.join(" + ")}</b></div></div>
    ${exp.error ? `<div class="message">${escapeHtml(exp.error)}</div>` : ""}
    <div style="overflow:auto"><table class="trial-table"><thead><tr><th>Trial</th><th>Status</th><th>Score</th>${metricNames.map(m => `<th>${pretty(m)}</th>`).join("")}<th>Duration</th></tr></thead><tbody>
      ${exp.trials.map(t => `<tr class="${t.number === exp.best_trial ? "best-row" : ""}"><td>#${t.number}${t.number === exp.best_trial ? " ★" : ""}</td><td>${t.status}</td><td>${t.score?.toFixed(4) ?? "—"}</td>${metricNames.map(m => `<td>${t.metrics[m]?.toFixed(4) ?? "—"}</td>`).join("")}<td>${t.duration_seconds ?? "—"}s</td></tr>`).join("")}
    </tbody></table></div>${["queued", "running"].includes(exp.status) ? `<button class="cancel" data-cancel="${exp.id}">Cancel after current trial</button>` : ""}`;
  $("#detail-modal").hidden = false;
  const cancel = $("[data-cancel]"); if (cancel) cancel.addEventListener("click", async () => { await api(`/api/experiments/${id}/cancel`, { method: "POST" }); $("#detail-modal").hidden = true; loadExperiments(); });
}

function escapeHtml(value) { const node = document.createElement("div"); node.textContent = value ?? ""; return node.innerHTML; }

async function init() {
  state.options = await api("/api/options");
  $("#yolo-version").innerHTML = Object.entries(state.options.versions).map(([value, info]) => `<option value="${value}">${info.label}</option>`).join("");
  $("#model-size").innerHTML = state.options.model_sizes.map(size => `<option value="${size}">${size.toUpperCase()} · ${{n:"Nano",s:"Small",m:"Medium",l:"Large",x:"Extra large"}[size]}</option>`).join("");
  updateTaskOptions(); renderSearchSpace(); renderParameterHelp(); await loadExperiments();
  $$('[name="task"]').forEach(input => input.addEventListener("change", updateTaskOptions));
  $("#yolo-version").addEventListener("change", updateResolvedModel); $("#model-size").addEventListener("change", updateResolvedModel);
  $$('[data-range-view]').forEach(button => button.addEventListener("click", () => switchRangeView(button.dataset.rangeView)));
  $("#inspect-dataset").addEventListener("click", inspectDataset); $("#experiment-form").addEventListener("submit", submitExperiment); $("#refresh").addEventListener("click", loadExperiments);
  $$('[data-close]').forEach(el => el.addEventListener("click", () => $("#detail-modal").hidden = true));
}
init().catch(error => { $("#form-message").textContent = error.message; $("#form-message").hidden = false; });

const state = { sourcePath: null, experiments: [], current: null, jobs: [], timer: null };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value ?? "";
  return node.innerHTML;
}

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—";
}

async function loadExperiments() {
  const message = $("#source-message");
  message.hidden = true;
  try {
    const result = await api("/api/training/experiments/inspect", {
      method: "POST",
      body: JSON.stringify({ path: $("#experiments-path").value })
    });
    state.sourcePath = result.path;
    state.experiments = result.experiments;
    $("#experiment-select").innerHTML = state.experiments.map(experiment =>
      `<option value="${experiment.id}">${escapeHtml(experiment.name)} · Trial ${experiment.best_trial}</option>`
    ).join("");
    $("#experiment-select-wrap").hidden = false;
    selectExperiment();
  } catch (error) {
    message.textContent = error.message;
    message.hidden = false;
    $("#experiment-select-wrap").hidden = true;
    $("#winner-preview").hidden = true;
    $("#training-form").hidden = true;
  }
}

function selectExperiment() {
  state.current = state.experiments.find(item => item.id === $("#experiment-select").value);
  if (!state.current) return;
  renderWinner();
  populateTrainingForm();
}

function renderWinner() {
  const experiment = state.current;
  const metrics = Object.entries(experiment.metrics);
  const hyperparameters = Object.entries(experiment.hyperparameters);
  const root = $("#winner-preview");
  root.innerHTML = `
    <div class="winner-header"><div><h3>${escapeHtml(experiment.name)}</h3><p>${experiment.model} · BEST TRIAL ${experiment.best_trial}</p></div><div class="winner-score"><span>OBJECTIVE SCORE</span><b>${formatMetric(experiment.score)}</b></div></div>
    <div class="metric-cards">${metrics.map(([name, value]) => `<div class="metric-card"><span>${escapeHtml(name.replaceAll("_", " "))}</span><b>${formatMetric(value)}</b></div>`).join("")}</div>
    <details class="hyper-details"><summary>Winning hyperparameters · ${hyperparameters.length} values</summary><div class="hyper-grid">${hyperparameters.map(([name, value]) => `<div><span>${escapeHtml(name)}</span><b title="${escapeHtml(value)}">${escapeHtml(value)}</b></div>`).join("")}</div></details>`;
  root.hidden = false;
}

function populateTrainingForm() {
  const experiment = state.current;
  const parsed = parseModel(experiment.model);
  $("#final-task").value = experiment.task;
  $('[name="version"]').value = parsed.version;
  $('[name="size"]').value = parsed.size;
  $('[name="batch"]').value = experiment.hyperparameters.batch || 16;
  $('[name="image_size"]').value = experiment.image_size || 640;
  $('[name="device"]').value = experiment.device ?? "";
  $("#final-dataset").value = experiment.dataset;
  $('[name="name"]').value = `${experiment.name} Final`;
  updateSplits();

  const canContinue = Boolean(experiment.last_weights);
  $("#continue-option").classList.toggle("disabled", !canContinue);
  $('[name="mode"][value="continue"]').disabled = !canContinue;
  $("#weight-warning").hidden = canContinue;
  if (!canContinue) $("#weight-warning").textContent = "Continue is unavailable because the winning trial's weights/last.pt was not found on this server. You can still start a new final run.";
  $('[name="mode"][value="new"]').checked = true;
  updateMode();
  $("#training-form").hidden = false;
}

function parseModel(model) {
  const match = String(model).match(/^(yolo26|yolo11|yolov8)([nsmlx])(?:-(?:seg|cls))?\.pt$/i);
  if (!match) return { version: "yolo26", size: "n" };
  return { version: match[1].toLowerCase(), size: match[2].trim().toLowerCase() };
}

function updateSplits() {
  const splits = state.current?.dataset_splits || {};
  $("#train-split").textContent = splits.train ?? "train";
  $("#val-split").textContent = splits.val ?? "val";
  $("#test-split").textContent = splits.test ?? "not configured";
}

function updateMode() {
  const continuing = $('[name="mode"]:checked').value === "continue";
  $("#new-training-fields").hidden = continuing;
  $("#continue-training-fields").hidden = !continuing;
  $(".primary span:first-child").textContent = continuing ? "Continue latest weights" : "Start final training";
}

async function submitTraining(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const mode = form.get("mode");
  const message = $("#training-message");
  const button = $(".primary", event.target);
  message.hidden = true;
  button.disabled = true;
  const body = {
    experiment_path: state.sourcePath,
    experiment_id: state.current.id,
    name: form.get("name"),
    mode,
    epochs: Number(mode === "continue" ? $("#continue-epochs").value : form.get("epochs")),
    dataset: mode === "continue" ? state.current.dataset : form.get("dataset"),
    task: mode === "continue" ? state.current.task : form.get("task"),
    version: form.get("version"),
    size: form.get("size"),
    batch: Number(mode === "continue" ? (state.current.hyperparameters.batch || 16) : form.get("batch")),
    image_size: Number(mode === "continue" ? state.current.image_size : form.get("image_size")),
    device: mode === "continue" ? state.current.device : (form.get("device") || null)
  };
  try {
    await api("/api/training/jobs", { method: "POST", body: JSON.stringify(body) });
    await loadJobs();
  } catch (error) {
    message.textContent = error.message;
    message.hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function loadJobs() {
  try {
    state.jobs = await api("/api/training/jobs");
    renderJobs();
  } catch (error) {
    $("#training-jobs").innerHTML = `<div class="message">${escapeHtml(error.message)}</div>`;
  }
  const active = state.jobs.some(job => ["queued", "running"].includes(job.status));
  clearTimeout(state.timer);
  if (active) state.timer = setTimeout(loadJobs, 2500);
}

function renderJobs() {
  const root = $("#training-jobs");
  if (!state.jobs.length) {
    root.innerHTML = '<div class="empty"><span>◇</span><p>No final runs yet</p><small>Created jobs will appear here.</small></div>';
    return;
  }
  root.innerHTML = state.jobs.map(job => `
    <article class="job-card"><div class="job-head"><b>${escapeHtml(job.name)}</b><span class="badge ${job.status}">${job.status}</span></div>
      <div class="job-meta">${job.mode.toUpperCase()} · ${job.model} · ${job.epochs} EPOCHS${job.resumed ? " · RESUMED" : ""}</div>
      ${Object.keys(job.metrics).length ? `<div class="job-metrics">${Object.entries(job.metrics).map(([name, value]) => `<span>${escapeHtml(name)} ${formatMetric(value)}</span>`).join("")}</div>` : ""}
      ${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ""}
      ${job.run_directory ? `<div class="job-path">${escapeHtml(job.run_directory)}/weights/best.pt</div>` : ""}
      ${job.status === "failed" ? `<button class="resume-job" data-resume="${job.id}">Resume from last checkpoint</button>` : ""}
    </article>`).join("");
  $$('[data-resume]').forEach(button => button.addEventListener("click", () => resumeJob(button.dataset.resume)));
}

async function resumeJob(id) {
  try { await api(`/api/training/jobs/${id}/resume`, { method: "POST" }); await loadJobs(); }
  catch (error) { window.alert(error.message); }
}

async function init() {
  $("#load-experiments").addEventListener("click", loadExperiments);
  $("#experiment-select").addEventListener("change", selectExperiment);
  $$('[name="mode"]').forEach(input => input.addEventListener("change", updateMode));
  $("#training-form").addEventListener("submit", submitTraining);
  $("#refresh-jobs").addEventListener("click", loadJobs);
  await loadJobs();
}

init();

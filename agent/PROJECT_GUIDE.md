# YOLO Hyperparameter Studio — Agent Development Guide

This document is the authoritative handoff for agents and developers continuing this repository. Read it together with `agent/SPEC.md`, but treat the current code and the invariants below as the source of truth when the original specification is outdated.

## 1. Project purpose

This is a local, single-machine web application for the complete YOLO experimentation workflow:

1. **Optimize** — run seeded random hyperparameter searches across Ultralytics YOLO detection, segmentation, or classification models.
2. **Train best** — import a completed optimizer result and train a full-budget final model using the winning hyperparameters.
3. **Validate** — compare one to twenty final `.pt` checkpoints on a held-out test split using `model.val()`.

The application intentionally uses no `argparse`, no frontend framework, and no external database server. It follows clean-architecture boundaries, uses FastAPI as the HTTP interface, persists local state in SQLite, and uses plain HTML/CSS/JavaScript for the frontend. Legacy JSON histories are supported through a one-time importer.

This is currently a practical MVP for a single user and normally one GPU. It is not a distributed training platform, multi-tenant service, or Optuna implementation.

## 2. Important terminology

- The optimization engine is **seeded random search**, not Optuna. Users may put “Optuna” in an experiment name, but the project does not import or invoke Optuna.
- A **trial** is one optimizer training run with one randomly sampled hyperparameter configuration.
- A **final-training job** is a full-budget training run created from the winner of an optimizer experiment.
- A **validation job** compares one or more `.pt` checkpoints on the YAML `test:` split.
- A **ticket** is a user-submitted feature request, bug report, or miscellaneous note stored for developers in SQLite.
- `best.pt` is the checkpoint with the best validation performance in a run.
- `last.pt` is the latest epoch checkpoint and is the checkpoint used for interruption recovery.
- `resume=True` restores a genuinely interrupted run, including epoch, optimizer, and scheduler state. Starting another training phase from a completed trial's `last.pt` is not the same operation; it loads those weights and starts a new phase.

## 3. Non-negotiable dataset-split invariant

Preserve this contract throughout all future development:

| Workflow | Weight updates | Evaluation used by workflow | Forbidden/reserved split |
| --- | --- | --- | --- |
| Optimize | YAML `train:` | YAML `val:` during `model.train()` | YAML `test:` |
| Train best | YAML `train:` | YAML `val:` during `model.train()` | YAML `test:` |
| Validate | None | YAML `test:` through `model.val(split="test")` | YAML `train:` and `val:` |

Do not silently fall back from `test` to `val` on the validation page. Do not use `test` for hyperparameter selection or final training. The whole point of the validation page is held-out evaluation.

For detection/segmentation, an ordinary dataset looks like:

```text
dataset/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: vehicle
```

The physical folder names are irrelevant to Ultralytics; the keys in `data.yaml` assign their roles. Labels are located through the usual images-to-labels path convention.

For classification, the final-training workflow requires a real `val`, `valid`, or `validation` directory so Ultralytics cannot fall back to `test`. Classification validation support should be handled carefully because it uses directory conventions rather than a detection-style data YAML.

## 4. Runtime and configuration

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`python main.py` loads root `config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 6769
  reload: true
```

The checked-in values may change. Never hardcode assumptions about the current port in new code. `infrastructure/configuration.py` validates this file. Running `uvicorn main:app` directly bypasses these YAML server settings and uses Uvicorn's CLI/default settings.

Environment variables:

- `HYPER_GUI_DATA` — SQLite persistence and artifact root; defaults to `<project>/data`.
- `HYPER_GUI_WORKERS` — background executor worker count; defaults to `1` to avoid GPU contention.

Stock model names may cause Ultralytics to download weights. Local `.pt` paths avoid this. The repository may contain locally downloaded `.pt` files; treat large weights as artifacts and do not casually modify or commit them.

## 5. Routes and pages

| Page | Route | Files |
| --- | --- | --- |
| Optimize | `/` | `frontend/index.html`, `styles.css`, `app.js` |
| Train best | `/training` | `frontend/training.html`, `training.css`, `training.js` |
| Validate | `/validation` | `frontend/validation.html`, `validation.css`, `validation.js` |

All three pages also load `frontend/ticket.css` and `frontend/ticket.js`. The header Ticket button opens a shared modal and submits to the ticket API.

Important API groups:

- `/api/options`
- `/api/datasets/inspect`
- `/api/experiments` and `/api/experiments/{id}`
- `/api/experiments/{id}/cancel`
- `/api/training/experiments/inspect`
- `/api/training/jobs` and `/api/training/jobs/{id}`
- `/api/training/jobs/{id}/resume`
- `/api/validation/jobs` and `/api/validation/jobs/{id}`
- `/api/validation/jobs/{id}/retry`
- `POST /api/tickets` (submission only; no public read endpoint)

`interfaces/api.py` is the composition root. It creates repositories, adapters, use cases, the background executor, recovery hooks, request models, and HTTP routes. Avoid putting business calculations there; route functions should validate/translate inputs and invoke application services.

## 6. Clean architecture and dependency direction

```text
domain/
    Framework-independent entities, enums, value objects, and naming functions.

application/
    Use cases and ports/protocols. May depend on domain. Must not import FastAPI
    or Ultralytics.

infrastructure/
    Adapters for Ultralytics, SQLite persistence, legacy JSON migration,
    YAML/filesystem inspection, and experiment importing. May depend on
    application/domain.

interfaces/
    FastAPI routes and dependency composition. Converts HTTP requests into
    domain/application calls.

frontend/
    Plain HTML/CSS/JavaScript. Talks to the HTTP API with fetch().

tests/
    Fast unit/integration-style tests, generally using fake Ultralytics modules
    and temporary files. Real training is never required for the test suite.
```

Dependency direction is inward:

```text
interfaces/infrastructure → application → domain
```

Do not import FastAPI, Pydantic, Ultralytics, filesystem repositories, or frontend concerns into `domain/`. Do not import Ultralytics into `application/`. Keep external imports lazy when they are heavyweight or optional, as the existing adapters do.

## 7. Optimization workflow

Key files:

- `domain/entities.py` — task types, status, search space, experiment config/results.
- `application/services.py` — random sampler, objective scoring, trial orchestration and recovery.
- `application/ports.py` — trainer/repository protocols.
- `infrastructure/yolo_trainer.py` — calls Ultralytics `YOLO(...).train()`.
- `infrastructure/sqlite.py` — active SQLite persistence and legacy JSON migration.
- `infrastructure/repository.py` — legacy JSON repository retained for migration/tests.
- `infrastructure/datasets.py` — dataset discovery and coarse validation.

Search behavior:

- Numeric values are sampled between configured `low` and `high`.
- `log=True` uses log-uniform sampling.
- Choice values such as `batch` and `optimizer` are selected randomly from lists.
- `random.Random(config.seed)` makes the sequence reproducible.
- Trials still differ from one another under a fixed seed. A fixed seed only reproduces the same sequence on another identical run.
- The same `seed` is passed into YOLO for every trial so hyperparameters, rather than unrelated training randomness, remain the main changing variable.
- Multiple selected objective metrics are combined by arithmetic mean.
- F1 is derived from precision and recall if necessary.
- A failed trial is recorded and later trials continue.

Default hyperparameters currently cover:

- Core optimizer/schedule: `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `warmup_momentum`.
- Loss weights: `box`, `cls`, `dfl`.
- Augmentation: `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `mosaic`, `mixup`, `copy_paste`.
- Choices: `batch`, `optimizer`.

The frontend's Parameter Help view explains these. Keep help text synchronized if parameters are added or removed.

Supported task metrics:

- Detection/segmentation: precision, recall, F1, mAP50, mAP50–95, fitness.
- Classification: top-1 accuracy, top-5 accuracy, fitness.

The GUI supports YOLO26, YOLO11, and YOLOv8 with N/S/M/L/X sizes. Model filenames are composed by version, scale, and task suffix.

## 8. Final-training workflow

Key files:

- `domain/training.py`
- `application/final_training.py`
- `infrastructure/experiment_importer.py`
- `infrastructure/final_trainer.py`
- `infrastructure/training_repository.py`

The page loads experiments with a valid `best_trial` directly from the active SQLite repository, exposes the winning metrics/hyperparameters, and checks whether the winning trial's `weights/last.pt` and `weights/best.pt` exist on the current server. No path input is required. The legacy JSON importer and inspection endpoint remain available for backward compatibility, but the current page does not use them.

There are two user-visible modes:

### New final run

- Starts from a user-selected pretrained YOLO `.pt` model name.
- Reuses the winning hyperparameters.
- Allows explicit task/version/size, epochs, batch, image size, dataset, and device controls.
- Explicit final-run controls override corresponding tuned values; for example the chosen batch replaces the winner's batch.

### Continue latest weights

- Loads the completed optimizer winner's `last.pt` as starting weights.
- Starts an additional training phase for the entered epoch count.
- It does not call `resume=True` on the already completed optimizer run.

### Interruption recovery

For either mode, once the new final job has produced its own `weights/last.pt`, a retry or application restart loads that checkpoint and calls:

```python
YOLO(last_pt).train(resume=True, val=True)
```

This restores the final job's epoch, optimizer, scheduler, and model state. If no checkpoint exists, the same job restarts from its original source and configuration.

Queued/running jobs are re-enqueued on application startup. A failed job can be manually resumed. Completed jobs are never rerun automatically.

## 9. Validation/comparison workflow

Key files:

- `domain/validation.py`
- `application/validation.py`
- `infrastructure/yolo_validator.py`
- `infrastructure/validation_repository.py`

The page accepts one to twenty distinct `.pt` paths and readable labels. It validates them sequentially using identical settings:

```python
model.val(
    data=dataset_yaml,
    split="test",
    conf=confidence,
    iou=iou,
    imgsz=image_size,
    batch=batch,
    plots=True,
)
```

Important semantics:

- The confidence threshold controls the minimum retained detection confidence. `0.001` is appropriate for standard mAP/PR-curve evaluation; a deployment threshold can be used to compare operational behavior.
- The IoU setting is the NMS overlap threshold. It is not the mAP50/mAP50–95 evaluation threshold.
- Validation batch affects memory use and throughput, not model training.
- The page supports one model even though the UI is designed for comparison.
- Completed model results are retained if a different model fails.
- Startup recovery skips completed model entries and continues unfinished ones.
- Retry resets only failed model entries.

Collected metrics can include precision, recall, F1, mAP50, mAP75, mAP50–95, mask equivalents, classification top-1/top-5 accuracy, fitness, and inference milliseconds. Per-class summaries are stored when Ultralytics exposes `summary()`.

The frontend draws charts directly on an HTML canvas. There is no charting dependency. If modifying charts, preserve high-DPI scaling and responsive resize behavior.

Per-class `Images` usually means the number of evaluated images containing at least one ground-truth instance of that class, while `Instances` is the total number of labeled objects for that class. The `all` row is useful for checking the total number of evaluated images.

## 10. Persistence and artifact layout

SQLite is the active relational metadata store under `HYPER_GUI_DATA` (default `data/`). There is no separate database server:

```text
data/
├── studio.db              # Active SQLite metadata database
├── experiments.json       # Preserved legacy migration source/backup
├── training_jobs.json     # Preserved legacy migration source/backup
├── validation_jobs.json   # Preserved legacy migration source/backup
├── runs/                  # Optimizer checkpoints and plots
├── final_runs/            # Final checkpoints and plots
└── validation_runs/       # model.val plots/artifacts
```

SQLite repositories use one connection per operation, foreign-key enforcement, WAL journal mode, a five-second busy timeout, short transactions, indexes for status/time, and hydration back into domain dataclasses/enums. Optimizer trials and validation models are child tables; flexible configs, metrics, hyperparameters, and per-class rows remain JSON payloads inside SQLite.

The `tickets` table stores `id`, `title`, `type` (`feature`, `bug`, or `misc`), `message`, originating `page`, `status`, and `created_at`. Ticket history is intentionally database-only for now. Developers can query it with `sqlite3 data/studio.db "SELECT * FROM tickets ORDER BY created_at DESC;"`.

`initialize_database()` atomically creates `studio.db.migrating`, imports any legacy JSON histories, runs integrity and foreign-key checks, renames the verified database into place, and leaves source JSON unchanged. Once `studio.db` exists, legacy JSON is no longer read or updated. See `migrate.md` for backup, verification, and rollback.

This remains a single-process/single-user design. SQLite is reliable for the workload, but the process-local executor and active-ID sets are not a distributed job system. Use PostgreSQL plus a real job queue if multi-process, multi-user, or multi-machine operation is introduced.

Do not edit a persisted schema without a versioned schema migration and backward-compatible hydration. Existing jobs and experiment files matter to users. Do not casually delete legacy JSON files; they are rollback backups.

## 11. Artifact naming and backward compatibility

Naming helpers live in `domain/naming.py`.

New optimizer folders:

```text
<experiment-name>-<short-experiment-id>-trial-<zero-padded-number>
Sprint11-MyJangum-3042ab38-trial-020
```

New final-training folders:

```text
<final-run-name>-<short-job-id>
Sprint11-MyJangum-Final-939e58e1
```

New validation folders:

```text
<comparison-name>-<short-id>-<model-number>-<model-label>
Sprint11-Comparison-f2b8a901-01-Final-v1
```

`safe_name()` retains alphanumerics, hyphens, and underscores; other characters become hyphens. Names are truncated to 60 characters before suffixes.

Do not rename existing folders. Old persisted optimizer trials and final jobs may use ID-first folder names. Recovery code deliberately falls back to those legacy names. Any future naming change must preserve old checkpoint discovery.

## 12. Frontend development style

The frontend deliberately uses no React, Vue, build system, npm dependencies, or template engine.

- HTML defines static structure.
- CSS defines all visual design.
- JavaScript fetches JSON APIs, creates dynamic content, polls jobs, and handles forms.
- Each page has page-specific JS/CSS, while `styles.css` supplies the shared design system.

Current branding has been user-customized to **Nicholas Ganteng** with an `N` mark. Preserve user customization unless explicitly asked to change it.

Shared CSS variables at the top of `frontend/styles.css` define main colors and radius. See root `custom_gui.md` for a beginner-oriented frontend customization guide.

Frontend conventions:

- Use `const $` and `const $$` helpers for local DOM queries.
- Use a small `api()` wrapper around `fetch()` and convert FastAPI `detail` errors into `Error` messages.
- Escape all user-controlled strings before inserting them into `innerHTML` using the page's `escapeHtml()` helper.
- Preserve IDs and `data-*` attributes used by JavaScript.
- Use `hidden` for mutually exclusive UI panels.
- Poll running jobs at approximately 2.5 seconds; stop polling when no job is active.
- Keep layouts responsive at roughly 980 px and 600–650 px breakpoints.
- Avoid external JS dependencies when a simple native implementation is reasonable.
- Every new page should be linked in the shared header navigation on all existing pages.

The CSS is compact and sometimes has multiple selectors on one line. New page-specific styles may be more conventionally formatted. Do not reformat the entire stylesheet unless asked; broad formatting creates noisy diffs and may overwrite user changes.

## 13. Backend coding style

- Use Python type annotations throughout.
- Prefer dataclasses for domain/persistence models and enums for constrained states.
- Use Pydantic models only at the HTTP boundary.
- Use `pathlib.Path`, not manual string path concatenation.
- Raise domain/input `ValueError` in infrastructure helpers and convert it to `HTTPException(400)` at the route boundary.
- Preserve external exception text in job/trial `error` fields so failures are diagnosable.
- Keep long-running YOLO work off request threads using the existing `ThreadPoolExecutor`.
- Default to sequential GPU work (`HYPER_GUI_WORKERS=1`). Parallel GPU jobs must be an explicit deployment decision.
- Lazily import Ultralytics inside adapters so the app/test suite can import without installing or initializing it.
- Keep pure calculations such as sampling, scoring, naming, and normalization independently testable.
- Use protocols for application-layer dependencies rather than concrete infrastructure types.
- Do not add `argparse`; server configuration belongs in `config.yaml`, and deployment/data settings may use explicitly documented environment variables.
- Do not silently broaden dataset use or substitute splits.

## 14. State and recovery rules

Statuses use `queued`, `running`, `completed`, `failed`, and `cancelled` where applicable.

### Optimizer

- Startup re-enqueues queued/running experiments.
- Completed and failed trials are skipped on recovery.
- A persisted running trial reuses its exact hyperparameters and folder name.
- If its `last.pt` exists, the YOLO adapter uses `resume=True`.
- User cancellation is checked between trials, not safely in the middle of an Ultralytics epoch.
- Failed experiments are not automatically restarted.

### Final training

- Startup re-enqueues queued/running jobs.
- Its own `last.pt` triggers a true resume.
- Failed jobs expose manual resume.
- Active in-memory ID sets prevent duplicate enqueueing in one process.

### Validation

- Startup re-enqueues queued/running validation jobs.
- Completed model entries are skipped.
- Failed entries are retained and can be selectively retried.
- Models run sequentially in a comparison.

If adding cancellation to final training or validation, do not claim it is immediate unless implemented through safe Ultralytics callbacks/process control. A thread cannot safely kill an in-progress training call.

## 15. Ultralytics integration conventions

Optimizer and training use:

```python
YOLO(model, task=task).train(...)
```

Hyperparameter dictionaries are expanded with `**hyperparameters`. Explicit run controls should be applied after deciding precedence and must not appear twice as keyword arguments.

True recovery uses:

```python
YOLO(last_pt, task=task).train(resume=True)
```

Validation uses:

```python
YOLO(model_path).val(split="test", ...)
```

Metric normalization lives in `infrastructure/yolo_trainer.py` and `infrastructure/yolo_validator.py`. Ultralytics uses task-specific result keys such as box `(B)` and mask `(M)`. When adding new tasks or metrics, support both official attributes and `results_dict` where practical, and add fake-result tests.

The application does not currently register Ultralytics callbacks. Therefore it tracks job/trial status but not live per-epoch progress, loss, ETA, or epoch metrics. If adding live progress, use official callbacks such as `on_train_epoch_end`, update persistence through an injected progress reporter, and ensure callback writes are thread-safe and not excessively frequent.

## 16. Testing and verification

Fast verification:

```bash
python -m compileall -q application domain infrastructure interfaces main.py tests
node --check frontend/app.js
node --check frontend/training.js
node --check frontend/validation.js
pytest -q
```

At the time this guide was created, all 22 tests passed. The test count will naturally increase.

Test philosophy:

- Never run real YOLO training in unit tests.
- Insert a fake `ultralytics` module into `sys.modules` for adapter tests.
- Verify exact arguments, especially `split="test"`, `val=True`, `conf`, `iou`, and `resume=True`.
- Use `tmp_path` for JSON repositories, YAML files, checkpoints, and output roots.
- Test persistence round-trips and legacy hydration when schemas/naming evolve.
- Test recovery: completed work must be skipped and interrupted work must retain configuration.
- Run JavaScript syntax checks after every frontend behavior change.

Relevant test files:

- `test_optimizer.py` — sampling, scoring, cancellation, recovery, naming.
- `test_datasets.py` — dataset discovery.
- `test_final_training.py` — importing winners, final adapter semantics, failure recovery.
- `test_validation.py` — test-split enforcement, metrics, comparison recovery.
- `test_configuration.py` — `config.yaml` server validation.
- `test_api_options.py` — exposed model/version/task options.

## 17. Security and operational considerations

This application accepts server-local filesystem paths from the browser and can start expensive GPU jobs. It is designed for trusted local/LAN use, not exposure to the public internet.

Before any public/multi-user deployment, add at least:

- Authentication and authorization.
- A restricted allowlist of dataset/model/result roots.
- CSRF/security review appropriate to deployment.
- Rate limits and quotas.
- Authentication-aware database access and versioned migrations appropriate to deployment.
- A process/job queue with GPU resource management.
- Structured logging and audit events.
- Safer file upload/selection rather than arbitrary server path entry.

`host: 0.0.0.0` exposes the service on all interfaces. Only use it on a trusted network with deliberate firewall configuration.

## 18. Known limitations and future improvements

- Despite user experiment labels, there is no Optuna/Bayesian optimization, pruning, or early-stopping search strategy.
- SQLite currently stores some flexible structures as JSON payload columns; frequently queried metrics may eventually deserve normalized/indexed columns.
- `ThreadPoolExecutor` jobs live in the web process. Multi-worker ASGI deployment would break the current single-process coordination assumptions.
- There is no live epoch progress because no Ultralytics callbacks are registered.
- Cancellation only exists between optimizer trials.
- Final hyperparameters selected with short tuning runs may not remain optimal at a much longer final epoch budget. A rigorous workflow should retest several top configurations at the intended budget.
- Hyperparameters tuned from pretrained weights should generally be applied to pretrained training. Scratch training should run a separate search using an architecture YAML.
- Confidence changes can affect reported precision/recall and operational comparison. Standard mAP comparison should normally retain a low confidence threshold.
- YOLO26 end-to-end behavior may make the NMS IoU argument less influential unless traditional NMS is enabled; follow current Ultralytics documentation when changing that behavior.
- No exports (ONNX/TensorRT/etc.) are currently managed by the GUI.
- No model registry, deletion UI, or artifact cleanup policy exists.

## 19. Safe development workflow for future agents

1. Read `agent/SPEC.md`, this guide, and the relevant feature files completely.
2. Inspect the worktree and preserve user changes, especially frontend branding and `config.yaml` values.
3. Identify which architectural layer owns the new behavior.
4. Update domain/application contracts before wiring infrastructure and routes.
5. Keep SQLite schema and legacy JSON migration changes backward-compatible.
6. Add or update the UI without breaking IDs consumed by JavaScript.
7. Add focused tests using fakes rather than real GPU work.
8. Run compilation, all JS syntax checks, and `pytest -q`.
9. Update `README.md`, `custom_gui.md`, and this guide when behavior or invariants change.
10. State clearly what was verified and what requires a real dataset/GPU.

Do not rename or delete user artifacts, JSON histories, checkpoints, datasets, or model weights without explicit authorization. Avoid changing unrelated user customization. Prefer additive, reversible changes and preserve restart recovery throughout.

## 20. Quick file map

```text
agent/
├── SPEC.md                    Original request
└── PROJECT_GUIDE.md           This agent handoff

domain/
├── entities.py                Optimizer entities/search space/status
├── training.py                Final-training job entity
├── validation.py              Validation job/model result entities
├── ticket.py                  User ticket entity and type enum
└── naming.py                  Readable and legacy-safe artifact naming

application/
├── ports.py                   Optimizer ports
├── services.py                Optimizer orchestration
├── final_training.py          Final-training orchestration
└── validation.py              Multi-model validation orchestration

infrastructure/
├── configuration.py           config.yaml loader
├── datasets.py                Dataset discovery
├── experiment_importer.py     Import experiments.json winner
├── sqlite.py                  Active schema, migration, SQLite repositories
├── repository.py              Legacy optimizer JSON repository/import helper
├── training_repository.py     Legacy final-training JSON repository/import helper
├── validation_repository.py   Legacy validation JSON repository/import helper
├── yolo_trainer.py            Optimizer Ultralytics adapter
├── final_trainer.py           Final-training Ultralytics adapter
└── yolo_validator.py          model.val comparison adapter

interfaces/
└── api.py                     FastAPI composition root and all routes

frontend/
├── index.html/styles.css/app.js
├── training.html/training.css/training.js
├── validation.html/validation.css/validation.js
└── ticket.css/ticket.js       Shared ticket popup used by every page

main.py                        ASGI export and config-driven launcher
config.yaml                    Host, port, reload
README.md                      User/developer overview
custom_gui.md                  Beginner frontend customization guide
migrate.md                     SQLite backup, migration, verification, rollback
```

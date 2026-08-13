# YOLO Hyperparameter Studio

A local web UI that runs randomized hyperparameter searches for Ultralytics YOLO detection, segmentation, and classification models. The GUI supports YOLO26, YOLO11, and YOLOv8 with Nano through Extra Large model sizes. It discovers dataset layouts, creates a YAML hyperparameter record for every trial, tracks metrics, and identifies the strongest trial using the metrics you select.

## Viability and scope

The project is a practical MVP for single-machine experiments. Trials run sequentially by default, which avoids multiple jobs fighting for the same GPU. Search, training, and validation state survives a server restart in SQLite at `data/studio.db`; Ultralytics artifacts and each generated `hyperparameters.yaml` remain under filesystem run directories. Existing JSON history is imported automatically on the first SQLite-enabled start and left untouched as a backup. See [`migrate.md`](migrate.md) before upgrading an existing installation.

The **Ticket** button in every page lets users submit a feature request, bug report, or miscellaneous note. Tickets are stored only in the `tickets` table in `data/studio.db`; there is intentionally no public ticket-history page or read API yet. A developer can inspect them with:

```bash
sqlite3 data/studio.db "SELECT created_at, type, title, message, page, status FROM tickets ORDER BY created_at DESC;"
```

Random search is a sound baseline and parallelizes naturally, but it does not learn from prior trials. For large training budgets, Optuna/Bayesian sampling, pruning, GPU scheduling, and a database-backed job queue would be valuable later additions. This app reports validation metrics emitted by Ultralytics. A truly untouched test set should be evaluated once after choosing a configuration, rather than used to tune the model.

The **Randomization ranges** section controls random search. Each trial samples a new value between the minimum and maximum for every numeric hyperparameter, and randomly selects one batch size and optimizer from their lists. A fixed random seed makes the sequence repeatable; it does not stop the values from being randomized. For example, rerunning seed `42` reproduces the same sequence of Trial 1, Trial 2, and Trial 3 samples, while those three trials still differ from one another. Use the **Parameter help** card in that section to replace the controls with plain-language explanations and the currently configured sampling range of every setting; switching back preserves the values you entered.

## Run

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open <http://127.0.0.1:8000>. No command-line argument parser is used; optional settings come from environment variables:

For a beginner-friendly explanation of how to change the logo, title, wording, colors, fonts, and layout, see [`custom_gui.md`](custom_gui.md).

- `HYPER_GUI_DATA`: result directory (default: `./data`)
- `HYPER_GUI_WORKERS`: simultaneous experiments (default: `1`)

### Server host and port

Edit the root `config.yaml` file to change the address used by `python main.py`:

```yaml
server:
  host: 127.0.0.1
  port: 8000
  reload: true
```

- `127.0.0.1` makes the GUI available only on the current computer.
- `0.0.0.0` listens on all network interfaces so another device can connect using this computer's LAN address. Only use it on a trusted network and configure firewall access deliberately.
- `port` must be between `1` and `65535` and must not already be occupied.
- `reload` should normally be `true` during development and `false` for deployment.

The configuration file is used by `python main.py`. Starting the server with the external `uvicorn main:app` command uses Uvicorn's CLI host and port instead.

The first use of a stock model such as `yolo11n.pt` may download its weights. Enter a local weights path to remain fully offline.

## Train a final model from optimizer results

Open <http://127.0.0.1:8000/training> or select **Train best** in the header.

1. The page loads completed optimizer experiments directly from the active SQLite `studio.db`.
2. Select a completed experiment. The page displays its winning trial, metrics, hyperparameters, dataset splits, and checkpoint availability. No `experiments.json` path is required.
3. Choose one of two methods:
   - **New final run** starts from a selected pretrained YOLO version/task/size and applies the winning hyperparameters. Epochs and batch size are explicit final-run controls.
   - **Continue latest weights** loads the winning trial's `weights/last.pt` and starts an additional training phase. Because the tuning run has already completed, this is intentionally a new phase from those weights rather than `resume=True` on the finished trial.
4. Final jobs and results appear on the right. Output is stored in `data/final_runs/<name>-<short-job-id>/`, with final weights under `weights/best.pt`.

The final-training adapter passes the dataset to `model.train(..., val=True)` and never selects the YAML `test` split. Therefore:

- the YAML `train:` entry updates model weights;
- the YAML `val:` entry validates during training;
- the YAML `test:` entry remains untouched for a separate final `model.val(split="test")` evaluation.

If final training is interrupted after at least one epoch, its own `weights/last.pt` is used with `resume=True`, restoring the optimizer, scheduler, and epoch. Queued/running final jobs recover automatically after an application restart. Failed jobs also have a manual **Resume from last checkpoint** action; if no checkpoint was written, the same job starts again from its original source and configuration.

## Validate and compare final models

Open <http://127.0.0.1:8000/validation> or select **Validate** in the header. This page performs held-out evaluation with Ultralytics `model.val()`:

Confidence and NMS IoU are exact numeric inputs from `0` through `1`, so values can be typed instead of approximated with sliders.

1. Enter a comparison name and a detection/segmentation dataset YAML that defines `test:`.
2. Choose between 1 and 20 models, give each a readable label, and paste each `.pt` checkpoint path.
3. Set confidence, NMS IoU, image size, batch size, and device.
4. Start validation. Models run sequentially so they do not compete for the same GPU.
5. Select a completed comparison to inspect the metric chart, full metric table, best values, and per-class results.

Every checkpoint is evaluated with the same arguments:

```python
model.val(
    data="/path/to/data.yaml",
    split="test",
    conf=0.001,
    iou=0.7,
    imgsz=640,
    batch=16,
    plots=True,
)
```

The page always uses the YAML `test:` split. It never evaluates on `train:` or `val:`. A low confidence such as `0.001` is recommended for standard mAP evaluation because it preserves the full precision-recall curve; increase it when comparing behavior at an operational detection threshold. The IoU control is the NMS overlap threshold, not the mAP evaluation IoU.

Validation state is stored in `data/studio.db`. Ultralytics plots and artifacts are saved with readable names under:

```text
data/validation_runs/<comparison-name>-<short-id>-<model-number>-<model-label>/
```

Queued and running comparisons recover after an application restart. Completed model results are skipped during recovery, and a partially failed comparison can retry only its failed checkpoints.

## Dataset layouts

Detection and segmentation accept either a dataset YAML path or a folder containing `data.yaml`, `dataset.yaml`, or another top-level `.yaml` file.

```text
vehicles/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Classification accepts a directory containing `train` and either `val` or `test`, with one subdirectory per class.

```text
animals/
├── train/{cats,dogs}/
├── val/{cats,dogs}/
└── test/{cats,dogs}/
```

## Architecture

```text
domain/          entities and validation; no framework dependencies
application/     optimizer use case and trainer/repository ports
infrastructure/  Ultralytics, filesystem datasets, SQLite, and migration adapters
interfaces/      FastAPI routes and dependency composition
frontend/        plain HTML, CSS, and JavaScript UI
tests/           fast unit tests with a fake trainer
```

The objective is the arithmetic mean of selected metrics. F1 is calculated from precision and recall if the trainer does not emit it directly. A failed trial is recorded without discarding successful trials. Cancellation takes effect before the next trial because Ultralytics training itself is not safely interruptible mid-epoch from this adapter.

### Interrupted-training recovery

Queued and running experiments are automatically re-enqueued when the application starts. Already completed trials are skipped. An interrupted trial keeps its original sampled hyperparameters and resumes from `data/runs/<experiment-name>-<short-experiment-id>-trial-<number>/weights/last.pt`, restoring its epoch, model weights, optimizer, and scheduler through Ultralytics. If interruption happened before the first checkpoint was written, that trial starts again from its original base model with the same hyperparameters. A user-cancelled or failed experiment is not automatically restarted.

## Test

```bash
pytest -q
```

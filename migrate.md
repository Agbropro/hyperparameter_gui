# Migrating project history from JSON to SQLite

The application now stores optimizer, final-training, and validation metadata in:

```text
data/studio.db
```

SQLite is built into Python, so there is no database server, username, password, or additional package to install. Model weights, images, plots, YAML files, and other large artifacts remain as normal filesystem files.

## What to back up before upgrading

Stop the application first. Do not start the upgraded version until these files have been copied somewhere safe.

### Required history files

Back up whichever of these exist under your configured data directory:

```text
data/experiments.json
data/training_jobs.json
data/validation_jobs.json
```

If `HYPER_GUI_DATA` is configured, replace `data/` with that directory.

These contain experiment settings, trial metrics, job status, hyperparameters, validation results, and paths to artifacts.

### Optional application artifacts

These directories are not part of the SQLite metadata migration and are not modified by it. You may back them up separately only if you also want a general disaster-recovery copy of generated checkpoints and plots:

```text
data/runs/
data/final_runs/
data/validation_runs/
```

They contain `best.pt`, `last.pt`, training plots, confusion matrices, prediction samples, and generated hyperparameter YAML files. The database stores their paths, not their binary contents.

### External datasets and checkpoints are outside migration scope

You do **not** need to back up or copy datasets and checkpoints located outside `HYPER_GUI_DATA` to perform this migration. For example:

```text
/mnt/secondary/.../data.yaml
/mnt/secondary/.../images/
/mnt/secondary/.../labels/
/mnt/secondary/.../weights/best.pt
```

The database only stores these paths as text. Migration reads neither the dataset contents nor external model contents, and it does not move, copy, modify, or delete them. Their independent backup policy belongs to the system that owns those files, not this migration.

### Application configuration

Back up custom project settings and UI edits if the source tree itself is being replaced:

```text
config.yaml
frontend/
custom_gui.md
```

The current frontend includes user-customized Nicholas Ganteng branding.

## Suggested backup command

After stopping the application, create a destination outside the project and copy the data directory. For example:

```bash
cp -a data /path/to/backup/hyperparameter-gui-data-before-sqlite
cp -a config.yaml /path/to/backup/config.yaml
```

Choose a real backup destination. Do not use the project directory as the only backup location.

For the metadata migration itself, the three legacy JSON files are the relevant backups. Artifact directories are optional and external data is out of scope.

## Automatic first-start migration

Start the upgraded application normally:

```bash
python main.py
```

When `studio.db` does not exist, startup performs this sequence:

1. Creates `studio.db.migrating` in the data directory.
2. Creates the SQLite schema.
3. Imports all records from the three legacy JSON files that exist.
4. Preserves experiment IDs, trial numbers, best-trial links, statuses, metrics, hyperparameters, timestamps, run names, and artifact paths.
5. Runs `PRAGMA integrity_check`.
6. Runs `PRAGMA foreign_key_check`.
7. Atomically renames the verified temporary database to `studio.db`.
8. Leaves every source JSON file unchanged.

If a JSON file is malformed or an integrity check fails, startup aborts, deletes the temporary database, and does not create `studio.db`.

Migration only happens automatically when `studio.db` is absent. Once it exists, SQLite is the source of truth and later manual edits to legacy JSON files are not imported.

## Files after successful migration

```text
data/
├── studio.db                 Active metadata database
├── experiments.json         Preserved legacy backup; no longer updated
├── training_jobs.json       Preserved legacy backup; no longer updated
├── validation_jobs.json     Preserved legacy backup; no longer updated
├── runs/                    Optimizer artifacts
├── final_runs/              Final-training artifacts
└── validation_runs/         Validation artifacts
```

Some legacy JSON files may not exist if that feature was never used. That is normal.

## Verify migration

First, open all three application pages and confirm their histories:

```text
/
/training
/validation
```

Then inspect database counts and integrity from the project directory:

```bash
python - <<'PY'
from pathlib import Path
from infrastructure.sqlite import database_summary

print(database_summary(Path("data/studio.db")))
PY
```

Expected output resembles:

```python
{
    'experiments': 2,
    'experiment_trials': 20,
    'training_jobs': 1,
    'validation_jobs': 1,
    'validation_models': 3,
    'integrity_check': 'ok',
    'foreign_key_violations': 0,
}
```

Counts will differ based on your history. The important values are:

```text
integrity_check: ok
foreign_key_violations: 0
```

The database stores migration source paths and imported top-level counts in its `schema_info` table.

## Current project migration result

During this upgrade, the existing project history was imported successfully:

```text
Experiments:        2
Experiment trials: 20
Training jobs:      0
Validation jobs:    0
Validation models:  0
Integrity check:    ok
Foreign-key errors: 0
```

The existing `data/experiments.json` was left unchanged.

## Backing up SQLite after migration

The easiest safe method is:

1. Stop the application.
2. Copy `data/studio.db`.
3. Optionally copy application artifact directories if you also want a broader disaster-recovery backup.

```bash
cp -a data/studio.db /path/to/backup/studio.db
```

When the application is running, SQLite may also have `studio.db-wal` and `studio.db-shm` files. Do not copy only `studio.db` from a live application and assume it is complete. Stop the application first or use SQLite's online backup API.

Example online backup while the application may be running:

```bash
python - <<'PY'
import sqlite3

source = sqlite3.connect("data/studio.db")
destination = sqlite3.connect("/path/to/backup/studio.db")
with destination:
    source.backup(destination)
destination.close()
source.close()
PY
```

Replace `/path/to/backup/studio.db` with a valid destination outside the project.

## Rollback

The upgraded code expects SQLite, so rollback requires both the former code version and the preserved JSON files.

1. Stop the application.
2. Move `studio.db`, `studio.db-wal`, and `studio.db-shm` aside if they exist.
3. Restore the pre-SQLite application code.
4. Restore the original JSON files to their original filenames.
5. Restore artifact directories if they were moved.
6. Start the old application.

Do not delete `studio.db` until you are confident the rollback works; it may contain new jobs created after migration that are absent from the legacy JSON backups.

## Important behavior after migration

- `studio.db` becomes the only active metadata source.
- Legacy JSON files remain static and are not kept synchronized.
- Deleting a database row does not automatically delete `.pt` files or plot directories.
- Moving artifact directories can break resume, continue-training, and validation paths even though historical metrics remain visible.
- The database stores references to model weights and datasets; those external files remain outside database and migration ownership.
- Existing evaluation results remain available because validation jobs and per-model result payloads are imported.
- Per-class metrics remain stored in each validation model's JSON payload inside SQLite.

## Troubleshooting

### Application says a legacy JSON file is invalid

Do not delete the file. Restore it from backup or repair its JSON syntax, confirm that its root value is an object keyed by IDs, remove any incomplete `studio.db.migrating`, and start again.

### History is empty after migration

Check whether `HYPER_GUI_DATA` points to a different directory. Migration reads JSON and writes `studio.db` under that configured directory, not necessarily the repository's `data/` folder.

### Historical metrics appear but resume fails

The database record exists, but the stored `last.pt` path does not. Restore the artifact directory or update the path through a future repair tool. Do not manually edit the SQLite binary file.

### Database is locked

The application enables WAL mode and a five-second busy timeout. A persistent lock commonly means another process is holding a write transaction or multiple application processes are running against the same database. Run only one application process for the current architecture.

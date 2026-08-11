"""Stable, filesystem-safe names for training artifacts."""

from __future__ import annotations


def safe_name(value: str, fallback: str = "run") -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:60] or fallback


def optimizer_run_name(experiment_name: str, experiment_id: str, trial_number: int) -> str:
    return f"{safe_name(experiment_name, 'experiment')}-{experiment_id[:8]}-trial-{trial_number:03d}"


def final_run_name(job_name: str, job_id: str) -> str:
    return f"{safe_name(job_name, 'final-model')}-{job_id[:8]}"

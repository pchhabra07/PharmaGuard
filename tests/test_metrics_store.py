"""
PharmaGuard — Metrics Store Unit Tests
==========================================
Tests the metrics_store module's load, save, and retrieval logic
using temporary JSON files.
"""

import json
from pathlib import Path

from src.tracking.metrics_store import (
    get_latest_run,
    load_metrics_history,
    save_run_metrics,
)


# ────────────────────────────────────────────────────────────────────
# load_metrics_history
# ────────────────────────────────────────────────────────────────────


class TestLoadMetricsHistory:
    """Test loading metrics from JSON files."""

    def test_load_existing_file(self, tmp_metrics_file):
        """Loading an existing metrics file should return a list of runs."""
        history = load_metrics_history(tmp_metrics_file)
        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["run_id"] == "run_001_abc123"

    def test_load_nonexistent_file(self, tmp_path):
        """Loading from a nonexistent path should return an empty list."""
        fake_path = tmp_path / "does_not_exist.json"
        history = load_metrics_history(fake_path)
        assert history == []

    def test_load_corrupt_json(self, tmp_path):
        """A corrupt JSON file should return an empty list (not crash)."""
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{ this is not valid json !!!", encoding="utf-8")
        history = load_metrics_history(bad_file)
        assert history == []


# ────────────────────────────────────────────────────────────────────
# save_run_metrics
# ────────────────────────────────────────────────────────────────────


class TestSaveRunMetrics:
    """Test saving new run metrics to the JSON store."""

    def test_save_creates_file(self, tmp_path, sample_config):
        """Saving to a new path should create the file and parent dirs."""
        metrics_path = tmp_path / "sub" / "dir" / "metrics.json"

        record = save_run_metrics(
            config=sample_config,
            threshold=0.24,
            metrics_train={"accuracy": 0.85, "f1": 0.84},
            metrics_val={"accuracy": 0.83, "f1": 0.82},
            metrics_test={"accuracy": 0.82, "f1": 0.81},
            version="v1",
            path=metrics_path,
        )

        assert metrics_path.exists()
        assert "run_id" in record
        assert record["threshold"] == 0.24
        assert record["version"] == "v1"

        # Verify the file contains valid JSON with the new record
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["run_id"] == record["run_id"]

    def test_save_appends_to_existing(self, tmp_metrics_file, sample_config):
        """Saving to an existing file should append (not overwrite)."""
        record = save_run_metrics(
            config=sample_config,
            threshold=0.30,
            metrics_train={"accuracy": 0.87},
            metrics_val={"accuracy": 0.85},
            metrics_test={"accuracy": 0.84},
            version="v2",
            path=tmp_metrics_file,
        )

        history = load_metrics_history(tmp_metrics_file)
        assert len(history) == 2  # Original + new
        assert history[1]["version"] == "v2"
        assert history[1]["threshold"] == 0.30

    def test_save_records_config_subset(self, tmp_path, sample_config):
        """The saved record should contain the training config subset."""
        metrics_path = tmp_path / "metrics.json"

        record = save_run_metrics(
            config=sample_config,
            threshold=0.24,
            metrics_train={"accuracy": 0.85},
            metrics_val={"accuracy": 0.83},
            metrics_test={"accuracy": 0.82},
            version="v1",
            path=metrics_path,
        )

        assert record["config"]["n_estimators"] == 500
        assert record["config"]["max_depth"] == 6
        assert record["config"]["learning_rate"] == 0.1


# ────────────────────────────────────────────────────────────────────
# get_latest_run
# ────────────────────────────────────────────────────────────────────


class TestGetLatestRun:
    """Test retrieving the most recent run."""

    def test_latest_from_single_run(self, tmp_metrics_file):
        """With one run, get_latest_run should return that run."""
        latest = get_latest_run(tmp_metrics_file)
        assert latest is not None
        assert latest["run_id"] == "run_001_abc123"

    def test_latest_from_empty(self, tmp_path):
        """With no runs, get_latest_run should return None."""
        fake_path = tmp_path / "empty.json"
        latest = get_latest_run(fake_path)
        assert latest is None

    def test_latest_returns_last_appended(self, tmp_metrics_file, sample_config):
        """After appending, get_latest_run should return the newest."""
        save_run_metrics(
            config=sample_config,
            threshold=0.30,
            metrics_train={"accuracy": 0.87},
            metrics_val={"accuracy": 0.85},
            metrics_test={"accuracy": 0.84},
            version="v2",
            path=tmp_metrics_file,
        )

        latest = get_latest_run(tmp_metrics_file)
        assert latest["version"] == "v2"

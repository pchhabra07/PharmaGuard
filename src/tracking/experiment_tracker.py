"""
PharmaGuard — MLflow Experiment Tracker
==========================================
Thin wrapper around MLflow that provides PharmaGuard-specific
convenience methods for logging training runs.

The tracker is **failure-tolerant**: if MLflow is disabled in config
or any logging call fails, the training pipeline continues unaffected.
All methods silently no-op when tracking is disabled.

Usage::

    tracker = ExperimentTracker(config)
    tracker.start_run("training-run-001")
    tracker.log_params(config["training"])
    tracker.log_metrics({"precision": 0.82, ...}, prefix="test")
    tracker.log_artifact("model_artifacts/xgb_model.json")
    tracker.end_run()
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow experiment tracker with graceful degradation.

    If ``config["mlflow"]["enabled"]`` is ``False`` or MLflow is not
    installed, every method becomes a silent no-op.  This keeps the
    training pipeline decoupled from the tracking infrastructure.

    Parameters
    ----------
    config : dict
        Parsed ``config.yaml``.  Reads the ``mlflow`` section for
        ``tracking_uri``, ``experiment_name``, and ``enabled``.
    """

    def __init__(self, config: dict):
        mlflow_cfg = config.get("mlflow", {})
        self.enabled = mlflow_cfg.get("enabled", False)
        self._run = None

        if not self.enabled:
            logger.info("MLflow tracking is DISABLED")
            return

        try:
            import mlflow  # noqa: F811

            self._mlflow = mlflow

            tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
            experiment_name = mlflow_cfg.get(
                "experiment_name", "PharmaGuard-ADR-Detection"
            )

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            logger.info(
                "MLflow initialized — URI: %s, Experiment: %s",
                tracking_uri,
                experiment_name,
            )

        except ImportError:
            logger.warning(
                "mlflow is not installed — tracking disabled. "
                "Install with: pip install mlflow"
            )
            self.enabled = False

        except Exception as exc:
            logger.warning("MLflow setup failed: %s — tracking disabled", exc)
            self.enabled = False

    # ── Run lifecycle ──────────────────────────────────────────────

    def start_run(self, run_name: str = None) -> None:
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Human-readable name for this run (e.g. ``"phase3-v1"``).
        """
        if not self.enabled:
            return
        try:
            self._run = self._mlflow.start_run(run_name=run_name)
            logger.info("MLflow run started: %s", run_name or "(unnamed)")
        except Exception as exc:
            logger.warning("Failed to start MLflow run: %s", exc)

    def end_run(self) -> None:
        """End the active MLflow run."""
        if not self.enabled:
            return
        try:
            self._mlflow.end_run()
            self._run = None
            logger.info("MLflow run ended")
        except Exception as exc:
            logger.warning("Failed to end MLflow run: %s", exc)

    # ── Parameter logging ──────────────────────────────────────────

    def log_params(self, params: dict, prefix: str = "") -> None:
        """
        Log a flat dictionary of parameters to MLflow.

        Non-scalar values (lists, dicts) are converted to strings.

        Parameters
        ----------
        params : dict
            Key-value pairs to log.
        prefix : str, optional
            Dot-separated prefix for each key (e.g. ``"training"``).
        """
        if not self.enabled:
            return
        try:
            for key, value in params.items():
                param_name = f"{prefix}.{key}" if prefix else str(key)
                self._mlflow.log_param(param_name, str(value))
        except Exception as exc:
            logger.warning("Failed to log params: %s", exc)

    # ── Metric logging ─────────────────────────────────────────────

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """
        Log a dictionary of numeric metrics to MLflow.

        Skips non-scalar entries like ``confusion_matrix``.

        Parameters
        ----------
        metrics : dict
            Metric name → float value.
        prefix : str, optional
            Prefix for metric names (e.g. ``"test"`` → ``"test.f1"``).
        """
        if not self.enabled:
            return
        try:
            for key, value in metrics.items():
                if key == "confusion_matrix":
                    continue
                if not isinstance(value, (int, float)):
                    continue
                metric_name = f"{prefix}.{key}" if prefix else str(key)
                self._mlflow.log_metric(metric_name, float(value))
        except Exception as exc:
            logger.warning("Failed to log metrics: %s", exc)

    def log_threshold(self, threshold: float, strategy: str) -> None:
        """
        Log the tuned decision threshold and strategy.

        Parameters
        ----------
        threshold : float
            Optimal threshold value.
        strategy : str
            Strategy used (e.g. ``"f1"``, ``"youden"``).
        """
        if not self.enabled:
            return
        try:
            self._mlflow.log_param("threshold_strategy", strategy)
            self._mlflow.log_metric("optimal_threshold", threshold)
        except Exception as exc:
            logger.warning("Failed to log threshold: %s", exc)

    # ── Artifact logging ───────────────────────────────────────────

    def log_artifact(self, path: Path) -> None:
        """
        Log a single file as an MLflow artifact.

        Parameters
        ----------
        path : Path
            Path to the file to log (model JSON, model card, etc.).
        """
        if not self.enabled:
            return
        try:
            path = Path(path)
            if path.exists():
                self._mlflow.log_artifact(str(path))
                logger.info("Logged artifact: %s", path.name)
            else:
                logger.warning("Artifact not found: %s", path)
        except Exception as exc:
            logger.warning("Failed to log artifact %s: %s", path, exc)

    def log_artifacts_dir(self, directory: Path, artifact_path: str = None) -> None:
        """
        Log all files in a directory as MLflow artifacts.

        Parameters
        ----------
        directory : Path
            Directory containing files to log.
        artifact_path : str, optional
            Sub-directory name in the MLflow artifact store.
        """
        if not self.enabled:
            return
        try:
            directory = Path(directory)
            if directory.exists() and directory.is_dir():
                self._mlflow.log_artifacts(str(directory), artifact_path)
                logger.info("Logged artifacts from: %s", directory)
            else:
                logger.warning("Artifact directory not found: %s", directory)
        except Exception as exc:
            logger.warning("Failed to log artifacts dir %s: %s", directory, exc)

"""
PharmaGuard — Configuration Validation Tests
=================================================
Tests that config.yaml exists and contains all required keys
needed by the application.
"""

from pathlib import Path

import yaml


# ── Project root ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class TestConfigFile:
    """Validate config.yaml structure and required keys."""

    def test_config_file_exists(self):
        """config.yaml must exist in the project root."""
        assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"

    def test_config_is_valid_yaml(self):
        """config.yaml must be parseable as valid YAML."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict), "config.yaml should be a YAML mapping"

    def test_config_has_faers_section(self):
        """config.yaml must contain the 'faers' section."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert "faers" in config
        assert "quarters" in config["faers"]
        assert isinstance(config["faers"]["quarters"], list)

    def test_config_has_paths_section(self):
        """config.yaml must contain the 'paths' section."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert "paths" in config
        assert "model_artifacts" in config["paths"]

    def test_config_has_preprocessing_section(self):
        """config.yaml must contain the 'preprocessing' section."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert "preprocessing" in config
        assert "target_column" in config["preprocessing"]
        assert "categorical_features" in config["preprocessing"]
        assert "numerical_features" in config["preprocessing"]

    def test_config_has_training_section(self):
        """config.yaml must contain the 'training' section with key hyperparams."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert "training" in config
        train = config["training"]
        required_keys = [
            "scale_pos_weight", "threshold", "n_estimators",
            "max_depth", "learning_rate", "eval_metric",
            "early_stopping_rounds", "threshold_strategy",
        ]
        for key in required_keys:
            assert key in train, f"Missing training key: {key}"

    def test_config_has_api_section(self):
        """config.yaml must contain the 'api' section."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert "api" in config
        assert "host" in config["api"]
        assert "port" in config["api"]
        assert "model_version" in config["api"]

    def test_config_has_mlflow_section(self):
        """config.yaml must contain the 'mlflow' section."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        assert "mlflow" in config
        assert "experiment_name" in config["mlflow"]

    def test_threshold_is_valid_range(self):
        """The decision threshold must be between 0 and 1."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        threshold = config["training"]["threshold"]
        assert 0.0 <= threshold <= 1.0, f"Threshold {threshold} out of range [0, 1]"

    def test_port_is_valid(self):
        """The API port must be a valid port number."""
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        port = config["api"]["port"]
        assert isinstance(port, int)
        assert 1 <= port <= 65535, f"Port {port} out of valid range"

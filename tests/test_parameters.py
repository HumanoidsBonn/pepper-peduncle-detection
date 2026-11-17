"""Unit tests for parameter validation that do not require a GPU."""

from pathlib import Path

import pytest

from pepper_peduncle_detection.inference import predict
from pepper_peduncle_detection.training import train


def test_predict_requires_existing_checkpoint(tmp_path: Path) -> None:
    """Inference should fail explicitly when the model is missing."""
    with pytest.raises(FileNotFoundError):
        predict(tmp_path / "missing.pt", "image.jpg", tmp_path / "out")


def test_train_requires_existing_dataset_config(tmp_path: Path) -> None:
    """Training should fail explicitly when the dataset YAML is missing."""
    with pytest.raises(FileNotFoundError):
        train(tmp_path / "missing.yaml", "yolov8l-seg.pt", tmp_path / "out")

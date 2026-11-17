"""Train a cropped pepper and peduncle instance-segmentation model."""

from pathlib import Path
from typing import Any


def train(
    data: Path,
    model_name: str,
    output_dir: Path,
    epochs: int = 150,
    image_size: int = 96,
    batch_size: int = 2048,
    device: str = "0",
    seed: int = 0,
    patience: int = 20,
) -> Any:
    """Train an Ultralytics segmentation model with explicit run parameters.

    Args:
        data: YOLO dataset YAML file.
        model_name: Initial checkpoint name or path.
        output_dir: Parent directory for the training run.
        epochs: Maximum number of training epochs.
        image_size: Square training resolution in pixels.
        batch_size: Number of images per batch.
        device: Ultralytics device selector.
        seed: Random seed.
        patience: Early-stopping patience in epochs.

    Returns:
        The Ultralytics training result.

    Raises:
        FileNotFoundError: If the dataset configuration does not exist.
        ValueError: If a positive integer parameter is not positive.
    """
    if not data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data}")
    positive_values = {
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "patience": patience,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    from ultralytics import YOLO

    model = YOLO(model_name)
    return model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        seed=seed,
        deterministic=True,
        patience=patience,
        val=True,
        save=True,
        project=str(output_dir),
        name="cropped-peduncle-yolov8l-seg",
    )

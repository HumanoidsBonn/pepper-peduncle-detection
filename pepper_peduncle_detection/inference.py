"""Run cropped pepper and peduncle instance-segmentation inference."""

from pathlib import Path
from typing import Any


def predict(
    model_path: Path,
    source: str,
    output_dir: Path,
    confidence: float = 0.5,
    iou: float = 0.3,
    image_size: int = 96,
    device: str | None = None,
) -> list[Any]:
    """Run the published crop model and save rendered predictions.

    Args:
        model_path: Path to an Ultralytics segmentation checkpoint.
        source: Image, directory, video, stream, or other Ultralytics source.
        output_dir: Directory in which predictions will be written.
        confidence: Minimum detection confidence.
        iou: Intersection-over-union threshold for non-maximum suppression.
        image_size: Square inference resolution in pixels.
        device: Ultralytics device selector, such as ``"0"`` or ``"cpu"``.

    Returns:
        Ultralytics result objects, one per processed image.

    Raises:
        FileNotFoundError: If the model checkpoint does not exist.
        ValueError: If a numeric inference parameter is outside its valid range.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("iou must be in [0, 1]")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    from ultralytics import YOLO

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device,
        retina_masks=True,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    return list(results)

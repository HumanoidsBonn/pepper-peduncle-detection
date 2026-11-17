"""Command-line entry point for cropped peduncle inference."""

import argparse
from pathlib import Path

from pepper_peduncle_detection.inference import predict


def main() -> None:
    """Parse command-line arguments and run inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--device")
    args = parser.parse_args()
    predict(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.image_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

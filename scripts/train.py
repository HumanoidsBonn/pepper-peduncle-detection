"""Command-line entry point for cropped peduncle model training."""

import argparse
from pathlib import Path

from pepper_peduncle_detection.training import train


def main() -> None:
    """Parse command-line arguments and start training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8l-seg.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    train(
        data=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

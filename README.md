# Pepper Peduncle Detection

This repository contains the cropped pepper and peduncle instance-segmentation
model used by the perception pipeline in **HortiBot: An Adaptive Multi-Arm
System for Robotic Horticulture of Sweet Peppers** (IROS 2024).

HortiBot first detects a pepper in the full camera image, expands its bounding
box, and applies this model to the resulting crop. Focusing on the fruit region
makes the small, visually inconspicuous peduncle easier to segment. The paper
reports a peduncle mask mAP@50 of 0.741, an F1 score of 0.781, and a combined
pepper/peduncle processing time of 100 ms on the HortiBot system.

## Resources

- [Dataset and trained checkpoint](https://huggingface.co/datasets/rohitmenon86/pepper-peduncle-detection)
- [HortiBot project](https://github.com/HumanoidsBonn/HortiBot)
- [Paper](https://doi.org/10.1109/IROS58592.2024.10802082)
- [Preprint](https://arxiv.org/abs/2403.15306)
- [Video](https://www.ais.uni-bonn.de/videos/IROS_2024_Lenz/)

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the trained checkpoint:

```bash
wget -O best.pt \
  https://huggingface.co/datasets/rohitmenon86/pepper-peduncle-detection/resolve/main/models/best.pt
```

## Inference on Cropped Images

The input may be one image, a directory, a video, or another source supported
by Ultralytics YOLO. Predictions are written to the requested output directory.

```bash
pepper-peduncle-predict \
  --model best.pt \
  --source /path/to/cropped/images \
  --output outputs/predictions
```

The class mapping is:

| ID | Class |
|---:|---|
| 0 | peduncle |
| 1 | pepper |

## Training

Download the Hugging Face dataset and edit the `path` field in
`configs/cropped-peduncle.yaml`, or pass another compatible YOLO segmentation
configuration. The published run used a pretrained YOLOv8L segmentation model,
96 px inputs, 150 epochs, and seed 0.

```bash
pepper-peduncle-train \
  --data configs/cropped-peduncle.yaml \
  --model yolov8l-seg.pt \
  --epochs 150 \
  --image-size 96 \
  --batch-size 2048 \
  --device 0
```

Large batch sizes require substantial GPU memory. Select a batch size suitable
for the available device. No automatic fallback is applied.

## Dataset

The paper release contains 94,077 training and 10,805 validation image/label
pairs. It combines derived crops from the Synthetic and Empirical *Capsicum
annuum* dataset, the Sweet Pepper and Peduncle Segmentation dataset, and BUP20.
See the Hugging Face dataset card for provenance, citations, and source-specific
licensing information.

## License

The code in this repository is released under AGPL-3.0 because it uses the
AGPL-licensed Ultralytics package. Dataset files retain their source-specific
terms; the repository license does not relicense them.

## Citation

```bibtex
@inproceedings{lenz2024hortibot,
  author={Lenz, Christian and Menon, Rohit and Schreiber, Michael and Jacob, Melvin Paul and Behnke, Sven and Bennewitz, Maren},
  title={{HortiBot}: An Adaptive Multi-Arm System for Robotic Horticulture of Sweet Peppers},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  pages={31--38},
  doi={10.1109/IROS58592.2024.10802082}
}
```

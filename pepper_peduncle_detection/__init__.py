"""Inference and training helpers for cropped pepper peduncle segmentation."""

from .inference import predict
from .training import train

__all__ = ["predict", "train"]

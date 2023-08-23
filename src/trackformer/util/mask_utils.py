"""
Plotting utilities to visualize image segmentations.
"""
import cv2
import numpy as np
from typing import Tuple


def mask_overlay(image, mask, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5):
    """
    Combines image and its segmentation mask into a single image.

    Params:
        image: Training image. np.ndarray, shape (H, W, 3)
        mask: Segmentation mask. np.ndarray, shape (H, W)
        color: RGB Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

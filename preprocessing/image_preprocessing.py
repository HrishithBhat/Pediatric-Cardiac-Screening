"""
Image Preprocessing Pipeline
==============================
Shared pipeline for both Ultrasound Still Images and Chest X-Rays.

Steps:
  1. Load image (grayscale or RGB)
  2. CLAHE contrast enhancement  (per channel for RGB, direct for grayscale)
  3. Aspect-ratio-preserving padding to target size
  4. Z-score normalisation (ImageNet statistics)

Returns a torch.Tensor of shape (3, H, W).

Dependencies: opencv-python, numpy, torch, torchvision
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# CLAHE Enhancement
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Parameters
    ----------
    image     : uint8 numpy array  (H, W) grayscale  OR  (H, W, 3) BGR/RGB
    clip_limit: threshold for contrast limiting
    tile_grid : size of the grid for histogram equalisation

    Returns
    -------
    np.ndarray  same dtype / shape as input, enhanced
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    if image.ndim == 2:
        return clahe.apply(image)

    # Process each channel independently in LAB space (better perceptual result)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge((l_ch, a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Aspect-Ratio-Preserving Pad & Resize
# ---------------------------------------------------------------------------

def aspect_preserving_pad(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    pad_value: int = 0,
) -> np.ndarray:
    """
    Resize image so that the larger dimension equals *target_size*,
    then pad the shorter dimension symmetrically with *pad_value*.

    Parameters
    ----------
    image       : (H, W) or (H, W, C)
    target_size : (target_H, target_W)

    Returns
    -------
    np.ndarray  shape exactly (target_H, target_W[, C])
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]

    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Pad to target
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    if image.ndim == 2:
        return cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(pad_value,) * image.shape[2]
    )


# ---------------------------------------------------------------------------
# Z-Score Normalisation
# ---------------------------------------------------------------------------

def zscore_normalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Normalise a (C, H, W) float tensor in [0,1]."""
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / (std_t + 1e-8)


# ---------------------------------------------------------------------------
# Full Image Preprocessor
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Generic image preprocessor for both ultrasound and X-ray modalities.

    Usage
    -----
    pre = ImagePreprocessor()
    tensor = pre("image.jpg")     # torch.Tensor  (3, 224, 224)
    tensor = pre(numpy_array)     # also accepts numpy arrays (H, W[, C])
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ):
        self.target_size = target_size
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.mean = mean
        self.std = std
        self.augment = augment

        self._aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if augment else None

    def _load(self, source) -> np.ndarray:
        if isinstance(source, str):
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return img  # BGR uint8
        elif isinstance(source, np.ndarray):
            return source
        elif isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))[:, :, ::-1].copy()  # RGB→BGR
        else:
            raise TypeError(f"Unsupported input type: {type(source)}")

    def __call__(self, source) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor  shape (3, target_H, target_W)  float32
        """
        img = self._load(source)

        # Ensure 3-channel BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # CLAHE
        img = apply_clahe(img, clip_limit=self.clahe_clip, tile_grid=self.clahe_tile)

        # Aspect-preserving pad
        img = aspect_preserving_pad(img, target_size=self.target_size)

        # Convert BGR uint8 → RGB float [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Optional augmentation (PIL-based transforms)
        if self.augment and self._aug_transform is not None:
            pil = TF.to_pil_image(tensor)
            pil = self._aug_transform(pil)
            tensor = TF.to_tensor(pil)

        # Z-score normalisation
        tensor = zscore_normalize(tensor, self.mean, self.std)
        return tensor  # (3, H, W)


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def get_ultrasound_preprocessor(augment: bool = False) -> ImagePreprocessor:
    return ImagePreprocessor(
        target_size=(224, 224),
        clahe_clip=2.0,
        clahe_tile=(8, 8),
        augment=augment,
    )


def get_xray_preprocessor(augment: bool = False) -> ImagePreprocessor:
    """X-rays benefit from slightly stronger CLAHE."""
    return ImagePreprocessor(
        target_size=(224, 224),
        clahe_clip=3.0,
        clahe_tile=(8, 8),
        augment=augment,
    )

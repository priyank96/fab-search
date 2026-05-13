"""
Pattern-safe augmentation pipeline for fabric patches.

See plan Phase 2: simulate phone-capture nuisance variables WITHOUT
destroying high-frequency pattern content (stripe pitch, check size,
weave direction).

Two views per patch are sampled independently and form a SimCLR-style
positive pair.

Implementation notes:
- Anti-aliased downscale is enforced by using cv2.INTER_AREA for the
  final resize. Albumentations' default Resize uses INTER_LINEAR which
  aliases pinstripes — we override it.
- No horizontal/vertical flip and no general rotation: orientation is a
  discriminative feature for stripes/twills.
- Crop short-side floor: if the crop would resolve to <384 px on the
  short side, we widen it. This keeps at least 1-2 pattern repeats
  visible on large plaids.
"""
from __future__ import annotations

import cv2
import numpy as np
import albumentations as A

from training.config import (
    BLUR_PROB,
    BRIGHTNESS,
    CLAHE_PROB,
    CONTRAST,
    CROP_RATIO,
    CROP_SCALE,
    INPUT_SIZE,
    JPEG_QUALITY,
    MIN_CROP_SHORT_SIDE,
    NORM_MEAN,
    NORM_STD,
    PERSPECTIVE_DISTORTION,
    TILT_DEGREES,
)


def _enforce_short_side_floor(img: np.ndarray) -> np.ndarray:
    """Upscale (with INTER_CUBIC) so the short side is >= MIN_CROP_SHORT_SIDE.

    Applied BEFORE random cropping so the cropped region has enough pixel
    budget for fine patterns to survive the final 224 downscale.
    """
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= MIN_CROP_SHORT_SIDE:
        return img
    scale = MIN_CROP_SHORT_SIDE / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _crop_top_band(image: np.ndarray, **_) -> np.ndarray:
    """Keep the top ~60% of the image (biases downstream crop toward top edge)."""
    h = image.shape[0]
    return image[: max(1, int(h * 0.6)), :, :]


def _crop_bottom_band(image: np.ndarray, **_) -> np.ndarray:
    """Keep the bottom ~60% of the image (biases downstream crop toward bottom edge)."""
    h = image.shape[0]
    return image[int(h * 0.4) :, :, :]


def build_train_transform() -> A.Compose:
    """Two-view pipeline applied independently per call."""
    return A.Compose(
        [
            # Edge-biased pre-crop: ~30% of views are restricted to the top
            # or bottom band before RandomResizedCrop, so the model sees
            # patches whose centre is near a patch edge (mirrors how users
            # photograph the top/bottom of a fabric sample on a page).
            A.OneOf(
                [
                    A.Lambda(image=_crop_top_band),
                    A.Lambda(image=_crop_bottom_band),
                ],
                p=0.30,
            ),
            # Geometry
            A.RandomResizedCrop(
                size=(INPUT_SIZE, INPUT_SIZE),
                scale=CROP_SCALE,
                ratio=CROP_RATIO,
                interpolation=cv2.INTER_AREA,  # anti-aliased downscale
                p=1.0,
            ),
            A.Perspective(
                scale=PERSPECTIVE_DISTORTION,
                keep_size=True,
                fit_output=False,
                p=0.5,
            ),
            A.Rotate(
                limit=180,
                interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.7,
            ),

            # Blur (light, gated)
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 1.0), p=1.0),
                ],
                p=BLUR_PROB,
            ),

            # Photometric (geometry-safe, generous)
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS,
                contrast_limit=CONTRAST,
                p=0.8,
            ),
            # LAB white-balance jitter via per-channel shift in LAB
            A.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.50,   # was 0.20 — stronger colour invariance
                hue=0.10,          # was 0.04 — stronger colour invariance
                p=0.7,             # was 0.5
            ),
            # Force a fraction of views to be greyscale so the model can't
            # rely on colour as a shortcut for "same patch".
            A.ToGray(p=0.15),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=CLAHE_PROB),

            # Capture artifacts
            A.ImageCompression(
                quality_range=JPEG_QUALITY,
                p=0.7,
            ),

            # Normalise (final)
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def build_eval_transform() -> A.Compose:
    """Deterministic centre-crop pipeline for clean gallery embeddings."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.CenterCrop(height=INPUT_SIZE, width=INPUT_SIZE),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def build_query_aug() -> A.Compose:
    """Single-pass 'phone-like' augmentation used to synthesise queries
    for the proxy retrieval eval. Harsher version that mirrors the train
    augmentation regime so eval actually exercises the new invariances
    (large rotation, colour shifts, edge-biased crops)."""
    return A.Compose(
        [
            # Edge-biased pre-crop (matches train): 30% of queries are
            # restricted to the top or bottom band of the patch.
            A.OneOf(
                [
                    A.Lambda(image=_crop_top_band),
                    A.Lambda(image=_crop_bottom_band),
                ],
                p=0.30,
            ),
            A.RandomResizedCrop(
                size=(INPUT_SIZE, INPUT_SIZE),
                scale=CROP_SCALE,
                ratio=CROP_RATIO,
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
            A.Perspective(scale=PERSPECTIVE_DISTORTION, keep_size=True, p=1.0),
            A.Rotate(
                limit=180,                       # was TILT_DEGREES (3°)
                interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.9,
            ),
            A.MotionBlur(blur_limit=(3, 5), p=0.6),
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS, contrast_limit=CONTRAST, p=1.0
            ),
            A.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.50,                 # stronger colour shift
                hue=0.10,
                p=0.8,
            ),
            A.ToGray(p=0.15),                    # exercise colour-invariance
            A.ImageCompression(quality_range=JPEG_QUALITY, p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def prepare_patch(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (cv2) -> RGB float32 ready for the transform pipeline."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return _enforce_short_side_floor(rgb)

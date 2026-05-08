"""
PyTorch Dataset and BatchSampler for fabric-patch contrastive training.

Dataset semantics:
- Each item returns two independently augmented views of the same patch
  (a SimCLR-style positive pair).
- For "single-pattern" tiles, the second view is sampled with 50%
  probability from a different quadrant of the same tile rather than
  the augmentation pipeline (free strong positive).

Batch sampler:
- Biases ~HARD_NEG_FRAC of each batch to share pdf_name with some
  anchor, surfacing same-catalog hard negatives in-batch.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from training.augment import build_eval_transform, build_train_transform, prepare_patch
from training.config import HARD_NEG_FRAC, REGION_PADDING_PX


def _crop_patch(page_path: str, x: int, y: int, w: int, h: int, pad: int = REGION_PADDING_PX) -> np.ndarray:
    """Crop (x,y,w,h) + padding from a cached page image. Returns BGR np array."""
    img = cv2.imread(page_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")
    H, W = img.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        # Degenerate bbox; fall back to full page rather than raising.
        return img
    return img[y0:y1, x0:x1].copy()


def _quadrant(img: np.ndarray, q: int) -> np.ndarray:
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    if q == 0:
        return img[:h2, :w2]
    if q == 1:
        return img[:h2, w2:]
    if q == 2:
        return img[h2:, :w2]
    return img[h2:, w2:]


class PatchDataset(Dataset):
    """Returns dict with view1, view2, pdf_name, idx."""

    def __init__(self, manifest: pd.DataFrame, train: bool = True, use_quadrant_pairs: bool = True):
        self.df = manifest.reset_index(drop=True)
        self.train = train
        self.use_quadrant_pairs = use_quadrant_pairs and train
        self.transform = build_train_transform() if train else build_eval_transform()

    def __len__(self) -> int:
        return len(self.df)

    def _to_tensor(self, img_rgb_norm: np.ndarray) -> torch.Tensor:
        # Albumentations Normalize returns float32 HWC.
        return torch.from_numpy(img_rgb_norm.transpose(2, 0, 1)).contiguous()

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            patch_bgr = _crop_patch(
                row["local_page"],
                int(row["tile_x"]), int(row["tile_y"]),
                int(row["tile_w"]), int(row["tile_h"]),
            )
        except FileNotFoundError:
            # Skip-by-substitute: sample a neighbour. Avoids killing the loader.
            return self.__getitem__((idx + 1) % len(self.df))

        single_pattern = bool(row.get("single_pattern", False))

        # View 1: always augmentation of the full patch.
        rgb = prepare_patch(patch_bgr)
        v1 = self.transform(image=rgb)["image"]

        # View 2: either another augmentation of full patch, OR a different
        # quadrant of the same tile (single-pattern free positive).
        use_quad = (
            self.use_quadrant_pairs
            and single_pattern
            and random.random() < 0.5
            and min(patch_bgr.shape[:2]) >= 64
        )
        if use_quad:
            q1, q2 = random.sample(range(4), 2)
            quad_bgr = _quadrant(patch_bgr, q2)
            rgb2 = prepare_patch(quad_bgr)
        else:
            rgb2 = rgb
        v2 = self.transform(image=rgb2)["image"]

        return {
            "view1": self._to_tensor(v1),
            "view2": self._to_tensor(v2),
            "pdf_name": str(row["pdf_name"]),
            "idx": idx,
        }


class HardNegativeBatchSampler(Sampler[list[int]]):
    """Yields batches biased to share pdf_name within a fraction of the batch.

    Strategy: for each batch, draw `n_groups` catalogs, allocate ~K samples
    per catalog summing to HARD_NEG_FRAC * batch_size, fill the rest with
    uniform-random patches from anywhere in the dataset.
    """

    def __init__(self, manifest: pd.DataFrame, batch_size: int, num_batches: int, seed: int = 0):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = random.Random(seed)
        # Group indices by pdf_name.
        self.by_catalog: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(manifest["pdf_name"].tolist()):
            self.by_catalog[name].append(i)
        self.all_indices = list(range(len(manifest)))
        # Sqrt-frequency catalog weights (see plan §"Further considerations").
        sizes = np.array([len(v) for v in self.by_catalog.values()], dtype=np.float64)
        weights = np.sqrt(sizes)
        weights /= weights.sum()
        self.catalog_names = list(self.by_catalog.keys())
        self.catalog_weights = weights

    def __len__(self) -> int:
        return self.num_batches

    def _sample_catalog(self) -> str:
        return self.rng.choices(self.catalog_names, weights=self.catalog_weights, k=1)[0]

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            n_hard = int(round(self.batch_size * HARD_NEG_FRAC))
            n_random = self.batch_size - n_hard
            batch: list[int] = []
            seen: set[int] = set()

            # Hard portion: groups of 4 from the same catalog -> in-batch
            # confusables.
            while len(batch) < n_hard:
                cat = self._sample_catalog()
                pool = self.by_catalog[cat]
                k = min(4, n_hard - len(batch), len(pool))
                if k <= 0:
                    continue
                picks = self.rng.sample(pool, k)
                for p in picks:
                    if p not in seen:
                        seen.add(p)
                        batch.append(p)

            # Random fill.
            while len(batch) < self.batch_size:
                p = self.rng.choice(self.all_indices)
                if p not in seen:
                    seen.add(p)
                    batch.append(p)

            yield batch

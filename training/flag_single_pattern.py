"""
Compute the 'single_pattern' flag for each tile in the manifest.

Mirrors src/tiler.py:67-102: split each patch into 4 quadrants, embed
all 4 with pretrained OpenCLIP, compute the minimum pairwise cosine
similarity, and flag the tile as single-pattern if min_sim >=
SIMILARITY_THRESHOLD (0.75).

Tiles flagged as single-pattern are eligible to yield 4-way positive
pairs during training (see training.dataset.PatchDataset).

This is a separate offline pass because it needs torch + open_clip
loaded, which build_dataset.py does not. Run it AFTER
build_dataset.py and BEFORE train_lora.py.
"""
from __future__ import annotations

import logging
from itertools import combinations

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.config import SIMILARITY_THRESHOLD
from training.config import EMB_DIM, MANIFEST_PATH, MODEL_NAME, PRETRAINED
from training.dataset import _crop_patch, _quadrant

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}. Run build_dataset.py first.")

    df = pd.read_parquet(MANIFEST_PATH)
    log.info("Loaded %d rows", len(df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading %s / %s on %s", MODEL_NAME, PRETRAINED, device)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    flags: list[bool] = []
    bad: int = 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="quadrant probe"):
            try:
                patch = _crop_patch(
                    row["local_page"],
                    int(row["tile_x"]), int(row["tile_y"]),
                    int(row["tile_w"]), int(row["tile_h"]),
                )
            except Exception:
                flags.append(False)
                bad += 1
                continue

            if min(patch.shape[:2]) < 64:
                flags.append(False)
                continue

            quads_pil = []
            for q in range(4):
                qimg = _quadrant(patch, q)
                rgb = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
                quads_pil.append(preprocess(Image.fromarray(rgb)))
            batch = torch.stack(quads_pil, dim=0).to(device)
            feats = model.encode_image(batch)
            feats = F.normalize(feats, dim=-1)
            sims = feats @ feats.T  # (4,4)
            # min pairwise (i<j)
            i, j = zip(*combinations(range(4), 2))
            min_sim = float(sims[list(i), list(j)].min().item())
            flags.append(min_sim >= SIMILARITY_THRESHOLD)

    df["single_pattern"] = pd.Series(flags, dtype=bool)
    df.to_parquet(MANIFEST_PATH, index=False)
    log.info("Updated %s — single_pattern=True on %d/%d (%.1f%%); bad rows: %d",
             MANIFEST_PATH,
             int(df["single_pattern"].sum()),
             len(df),
             100.0 * df["single_pattern"].mean(),
             bad)


if __name__ == "__main__":
    main()

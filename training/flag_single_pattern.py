"""
Compute the 'single_pattern' flag for each tile in the manifest.

Mirrors src/tiler.py:67-102: split each patch into 4 quadrants, embed
all 4 with pretrained OpenCLIP, compute the minimum pairwise cosine
similarity, and flag the tile as single-pattern if min_sim >=
SIMILARITY_THRESHOLD (0.75).

Tiles flagged as single-pattern are eligible to yield 4-way positive
pairs during training (see training.dataset.PatchDataset).

Run AFTER build_dataset.py and BEFORE train_lora.py.

Performance:
- Tiles are batched into groups of TILE_BATCH (env: FLAG_TILE_BATCH)
  so the GPU sees TILE_BATCH * 4 images per forward pass.
- Page images are decoded in a worker pool (env: FLAG_DECODE_WORKERS)
  to keep the GPU fed.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
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
from training.config import MANIFEST_PATH, MODEL_NAME, PRETRAINED
from training.dataset import _crop_patch, _quadrant

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TILE_BATCH = int(os.environ.get("FLAG_TILE_BATCH", "32"))   # tiles per GPU batch (=> 4× images per forward)
DECODE_WORKERS = int(os.environ.get("FLAG_DECODE_WORKERS", "4"))
MIN_SIDE = 64

# Precompute pairwise indices once (6 unique upper-triangular pairs of 4 items).
_PAIR_I, _PAIR_J = zip(*combinations(range(4), 2))


def _prep_tile(args):
    """Decode page, crop tile, return preprocessed quadrant tensors. Runs in a thread.
    Returns ('ok', row_idx, [4 tensors]) or ('bad'|'small', row_idx).
    """
    row_idx, row, preprocess = args
    try:
        patch = _crop_patch(
            row["local_page"],
            int(row["tile_x"]), int(row["tile_y"]),
            int(row["tile_w"]), int(row["tile_h"]),
        )
    except Exception:
        return ("bad", row_idx)
    if min(patch.shape[:2]) < MIN_SIDE:
        return ("small", row_idx)
    try:
        quads = []
        for q in range(4):
            qimg = _quadrant(patch, q)
            rgb = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
            quads.append(preprocess(Image.fromarray(rgb)))
        return ("ok", row_idx, quads)
    except Exception:
        return ("bad", row_idx)


def _flush(pending, model, device, use_amp, amp_dtype, pair_i, pair_j, flags):
    """Encode a batch of N tiles (= 4N images) and update flags in-place."""
    n_tiles = len(pending)
    images = torch.stack([t for (_, quads) in pending for t in quads], dim=0)
    images = images.to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        feats = model.encode_image(images)
    feats = F.normalize(feats.float(), dim=-1)
    feats = feats.view(n_tiles, 4, -1)              # (N, 4, D)
    sims = torch.bmm(feats, feats.transpose(1, 2))  # (N, 4, 4)
    pair_sims = sims[:, pair_i, pair_j]             # (N, 6)
    min_sims = pair_sims.min(dim=1).values          # (N,)
    flags_batch = (min_sims >= SIMILARITY_THRESHOLD).cpu().numpy()
    for (row_idx, _), is_sp in zip(pending, flags_batch):
        flags[row_idx] = bool(is_sp)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}. Run build_dataset.py first.")

    df = pd.read_parquet(MANIFEST_PATH)
    log.info("Loaded %d rows", len(df))
    log.info("Config: tile_batch=%d, decode_workers=%d", TILE_BATCH, DECODE_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading %s / %s on %s", MODEL_NAME, PRETRAINED, device)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    flags = np.zeros(len(df), dtype=bool)
    bad = 0
    small = 0

    rows = [(i, df.iloc[i], preprocess) for i in range(len(df))]
    pair_i = torch.tensor(_PAIR_I, dtype=torch.long, device=device)
    pair_j = torch.tensor(_PAIR_J, dtype=torch.long, device=device)

    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool, torch.no_grad():
        pending: list[tuple[int, list]] = []
        pbar = tqdm(total=len(rows), desc="quadrant probe")
        for result in pool.map(_prep_tile, rows, chunksize=8):
            tag = result[0]
            if tag == "bad":
                bad += 1
                pbar.update(1)
                continue
            if tag == "small":
                small += 1
                pbar.update(1)
                continue
            _, row_idx, quads = result
            pending.append((row_idx, quads))
            if len(pending) >= TILE_BATCH:
                _flush(pending, model, device, use_amp, amp_dtype, pair_i, pair_j, flags)
                pbar.update(len(pending))
                pending.clear()
        if pending:
            _flush(pending, model, device, use_amp, amp_dtype, pair_i, pair_j, flags)
            pbar.update(len(pending))
            pending.clear()
        pbar.close()

    df["single_pattern"] = pd.Series(flags, dtype=bool)
    df.to_parquet(MANIFEST_PATH, index=False)
    log.info(
        "Updated %s — single_pattern=True on %d/%d (%.1f%%); bad: %d, too small: %d",
        MANIFEST_PATH,
        int(df["single_pattern"].sum()),
        len(df),
        100.0 * df["single_pattern"].mean(),
        bad,
        small,
    )


if __name__ == "__main__":
    main()

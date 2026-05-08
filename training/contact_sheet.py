"""
Render contact sheets to visually verify augmentations don't destroy
fine-stripe / small-check pattern detail BEFORE spending GPU.

Run:
    python -m training.contact_sheet --kind aug
    python -m training.contact_sheet --kind quad
    python -m training.contact_sheet --kind fine    # 50 finest-stripe + 50 smallest-check (best-effort heuristic)
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from training.augment import build_query_aug, build_train_transform, prepare_patch
from training.config import CONTACT_SHEET_DIR, INPUT_SIZE, MANIFEST_PATH, NORM_MEAN, NORM_STD
from training.dataset import _crop_patch, _quadrant

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _denorm(img_norm: np.ndarray) -> np.ndarray:
    out = img_norm * np.array(NORM_STD) + np.array(NORM_MEAN)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _grid(images: list[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 32, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _augmentation_sheet(df: pd.DataFrame, n: int, out_path: Path) -> None:
    transform = build_train_transform()
    panels: list[np.ndarray] = []
    df = df.sample(n=min(n, len(df)), random_state=0).reset_index(drop=True)
    for _, row in df.iterrows():
        try:
            patch = _crop_patch(row["local_page"], int(row["tile_x"]), int(row["tile_y"]),
                                int(row["tile_w"]), int(row["tile_h"]))
        except Exception:
            continue
        rgb = prepare_patch(patch)
        # Original (centre crop) for reference.
        h, w = rgb.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        orig = rgb[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]
        orig = cv2.resize(orig, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        v1 = _denorm(transform(image=rgb)["image"])
        v2 = _denorm(transform(image=rgb)["image"])
        triplet = np.concatenate([
            _label(orig, "orig"),
            _label(v1, "view1"),
            _label(v2, "view2"),
        ], axis=1)
        panels.append(triplet)
    sheet = _grid(panels, cols=2)
    sheet = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), sheet)
    log.info("Wrote %s", out_path)


def _quadrant_sheet(df: pd.DataFrame, n: int, out_path: Path) -> None:
    sp = df[df.get("single_pattern", False) == True]  # noqa: E712
    if len(sp) == 0:
        log.warning("No single_pattern tiles in manifest; run flag_single_pattern.py first.")
        return
    sp = sp.sample(n=min(n, len(sp)), random_state=0).reset_index(drop=True)
    panels = []
    for _, row in sp.iterrows():
        try:
            patch = _crop_patch(row["local_page"], int(row["tile_x"]), int(row["tile_y"]),
                                int(row["tile_w"]), int(row["tile_h"]))
        except Exception:
            continue
        if min(patch.shape[:2]) < 64:
            continue
        quads = []
        for q in range(4):
            qimg = _quadrant(patch, q)
            rgb = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            quads.append(_label(rgb, f"q{q}"))
        panels.append(np.concatenate(quads, axis=1))
    sheet = _grid(panels, cols=1)
    sheet = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), sheet)
    log.info("Wrote %s", out_path)


def _fine_pattern_sheet(df: pd.DataFrame, n: int, out_path: Path) -> None:
    """Heuristic: 'fine' = high spatial-frequency content via Laplacian variance.

    No labels exist, so we proxy fine stripes/small checks by patches whose
    Laplacian variance is in the top quantile (after normalising by patch
    area). Sample ~n from the top 5% as a stand-in.
    """
    scored: list[tuple[float, int]] = []
    sub = df.sample(n=min(2000, len(df)), random_state=1).reset_index(drop=True)
    for i, row in sub.iterrows():
        try:
            patch = _crop_patch(row["local_page"], int(row["tile_x"]), int(row["tile_y"]),
                                int(row["tile_w"]), int(row["tile_h"]))
        except Exception:
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Downsample so the metric is scale-comparable.
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        scored.append((var, i))
    scored.sort(reverse=True)
    pick_idx = [i for _, i in scored[:n]]
    fine_df = sub.iloc[pick_idx].reset_index(drop=True)
    _augmentation_sheet(fine_df, n=len(fine_df), out_path=out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("aug", "quad", "fine"), default="aug")
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args()

    df = pd.read_parquet(MANIFEST_PATH)
    if "single_pattern" in df.columns:
        df["single_pattern"] = df["single_pattern"].fillna(False).astype(bool)

    out_path = CONTACT_SHEET_DIR / f"contact_{args.kind}.png"
    if args.kind == "aug":
        _augmentation_sheet(df, args.n, out_path)
    elif args.kind == "quad":
        _quadrant_sheet(df, args.n, out_path)
    else:
        _fine_pattern_sheet(df, n=max(args.n, 50), out_path=out_path)


if __name__ == "__main__":
    main()

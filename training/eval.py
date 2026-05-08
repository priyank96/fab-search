"""
Evaluation probes for FabSearch fine-tuning.

- run_proxy_eval: Recall@1/5/10 on val patches with augmented queries.
- run_quadrant_probe: intra-tile vs cross-tile cosine on single-pattern tiles.
- run_pattern_probe: per-bucket Recall@5 and within/across cosine gap, given
  a hand-curated bucket CSV (id,bucket).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.augment import build_eval_transform, build_query_aug, prepare_patch
from training.config import MANIFEST_PATH
from training.dataset import _crop_patch, _quadrant

log = logging.getLogger(__name__)


def _to_tensor(img_norm_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_norm_hwc.transpose(2, 0, 1)).contiguous()


@torch.no_grad()
def _encode_patches(
    encoder, df: pd.DataFrame, device: torch.device, transform, batch_size: int = 64,
) -> tuple[torch.Tensor, list[int]]:
    encoder.eval()
    feats: list[torch.Tensor] = []
    kept_idx: list[int] = []
    buf: list[torch.Tensor] = []
    buf_idx: list[int] = []

    def _flush():
        if not buf:
            return
        x = torch.stack(buf, dim=0).to(device, non_blocking=True)
        z = encoder(x).float().cpu()
        feats.append(z)
        kept_idx.extend(buf_idx)
        buf.clear()
        buf_idx.clear()

    for i, row in df.reset_index(drop=True).iterrows():
        try:
            patch = _crop_patch(
                row["local_page"], int(row["tile_x"]), int(row["tile_y"]),
                int(row["tile_w"]), int(row["tile_h"]),
            )
        except Exception:
            continue
        rgb = prepare_patch(patch)
        out = transform(image=rgb)["image"]
        buf.append(_to_tensor(out))
        buf_idx.append(i)
        if len(buf) >= batch_size:
            _flush()
    _flush()

    if not feats:
        return torch.empty(0, 512), []
    return torch.cat(feats, dim=0), kept_idx


@torch.no_grad()
def run_proxy_eval(encoder, val_df: pd.DataFrame, device: torch.device) -> dict[str, float]:
    """Augmented-query retrieval over the val split.

    Gallery: each val patch encoded with the deterministic eval transform.
    Queries: same patches encoded with a strong phone-like augmentation.
    Targets: gallery index of the same patch. Reports Recall@1/5/10.
    """
    if len(val_df) == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "n": 0}

    eval_t = build_eval_transform()
    query_t = build_query_aug()

    gallery, gallery_idx = _encode_patches(encoder, val_df, device, eval_t)
    queries, query_idx = _encode_patches(encoder, val_df, device, query_t)
    if gallery.numel() == 0 or queries.numel() == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "n": 0}

    # Map idx -> gallery row.
    pos = {gi: r for r, gi in enumerate(gallery_idx)}
    sims = queries @ gallery.T  # (Q, G)
    # Mask out the query's exact gallery row from being a "trivial" hit?
    # The gallery and query are different views (clean vs augmented) so we
    # WANT the same row to match. No masking.

    ranks = []
    for qi, qrow in enumerate(query_idx):
        if qrow not in pos:
            continue
        target = pos[qrow]
        order = torch.argsort(sims[qi], descending=True)
        rank = int((order == target).nonzero(as_tuple=True)[0].item()) + 1
        ranks.append(rank)
    if not ranks:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "n": 0}
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    return {
        "recall@1": float((ranks_t <= 1).float().mean().item()),
        "recall@5": float((ranks_t <= 5).float().mean().item()),
        "recall@10": float((ranks_t <= 10).float().mean().item()),
        "median_rank": float(ranks_t.median().item()),
        "n": len(ranks),
    }


@torch.no_grad()
def run_quadrant_probe(
    encoder, val_df: pd.DataFrame, device: torch.device, max_tiles: int = 200,
) -> dict[str, float]:
    """Intra-tile vs cross-tile cosine on single-pattern tiles."""
    sp = val_df[val_df.get("single_pattern", False) == True].reset_index(drop=True)  # noqa: E712
    if len(sp) == 0:
        return {"intra_mean": 0.0, "inter_mean": 0.0, "gap": 0.0, "n_tiles": 0}
    sp = sp.head(max_tiles)
    eval_t = build_eval_transform()

    encoder.eval()
    tile_quad_feats: list[torch.Tensor] = []  # each (4, D)
    for _, row in tqdm(sp.iterrows(), total=len(sp), desc="quad probe"):
        try:
            patch = _crop_patch(
                row["local_page"], int(row["tile_x"]), int(row["tile_y"]),
                int(row["tile_w"]), int(row["tile_h"]),
            )
        except Exception:
            continue
        if min(patch.shape[:2]) < 64:
            continue
        quads = []
        for q in range(4):
            qimg = _quadrant(patch, q)
            rgb = prepare_patch(qimg)
            quads.append(_to_tensor(eval_t(image=rgb)["image"]))
        x = torch.stack(quads, dim=0).to(device)
        z = encoder(x).float().cpu()
        tile_quad_feats.append(z)

    if not tile_quad_feats:
        return {"intra_mean": 0.0, "inter_mean": 0.0, "gap": 0.0, "n_tiles": 0}

    intra = []
    for z in tile_quad_feats:
        sims = z @ z.T
        n = sims.size(0)
        iu = torch.triu_indices(n, n, offset=1)
        intra.extend(sims[iu[0], iu[1]].tolist())

    # inter: pick a random quadrant from each tile, compute pairwise.
    rep = torch.stack([z[0] for z in tile_quad_feats], dim=0)
    sims = rep @ rep.T
    n = sims.size(0)
    iu = torch.triu_indices(n, n, offset=1)
    inter = sims[iu[0], iu[1]].tolist()

    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "gap": intra_mean - inter_mean,
        "n_tiles": len(tile_quad_feats),
    }


@torch.no_grad()
def run_pattern_probe(
    encoder, val_df: pd.DataFrame, device: torch.device, buckets_csv: Path,
) -> dict[str, float]:
    """Per-bucket Recall@5 and within/across cosine gap.

    buckets_csv: CSV with columns `id,bucket` mapping vector ID to one of
    {fine_stripe, wide_stripe, small_check, large_check, plain, floral}.
    Hand-curated, see docs in plan Phase 4 §16.
    """
    if not buckets_csv.exists():
        log.warning("Pattern bucket CSV missing: %s — skipping pattern probe", buckets_csv)
        return {}

    buckets = pd.read_csv(buckets_csv)
    df = val_df.merge(buckets, on="id", how="inner")
    if len(df) == 0:
        log.warning("No overlap between val_df and buckets_csv ids")
        return {}

    eval_t = build_eval_transform()
    feats, kept = _encode_patches(encoder, df, device, eval_t)
    if feats.numel() == 0:
        return {}
    df_kept = df.iloc[kept].reset_index(drop=True)

    sims = feats @ feats.T
    sims.fill_diagonal_(-1.0)

    out: dict[str, float] = {}
    for bucket, grp in df_kept.groupby("bucket"):
        rows = grp.index.to_numpy()
        if len(rows) < 2:
            continue
        # Recall@5 restricted: target = nearest within same bucket.
        hits = 0
        for r in rows:
            top5 = torch.topk(sims[r], k=5).indices.tolist()
            same_bucket = [df_kept.iloc[t]["bucket"] == bucket for t in top5]
            if any(same_bucket):
                hits += 1
        out[f"{bucket}/recall@5"] = hits / len(rows)

    # Within vs across cosine.
    bucket_arr = df_kept["bucket"].to_numpy()
    same = bucket_arr[:, None] == bucket_arr[None, :]
    np.fill_diagonal(same, False)
    sims_np = sims.numpy()
    within = sims_np[same].mean() if same.any() else 0.0
    across = sims_np[~same & ~np.eye(len(sims_np), dtype=bool)].mean()
    out["pattern/within_mean"] = float(within)
    out["pattern/across_mean"] = float(across)
    out["pattern/gap"] = float(within - across)
    return out


def main() -> None:
    """Standalone eval: load best checkpoint and print all metrics."""
    import argparse
    from training.config import CHECKPOINT_DIR
    from training.model import build_model

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "best.pt"))
    p.add_argument("--buckets", default=str(Path(__file__).parent / "pattern_buckets.csv"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = build_model(device)
    if Path(args.ckpt).exists():
        state = torch.load(args.ckpt, map_location=device)
        encoder.load_state_dict(state["encoder_state"], strict=False)
        log.info("Loaded checkpoint %s", args.ckpt)
    else:
        log.warning("No checkpoint at %s; evaluating pretrained baseline.", args.ckpt)

    df = pd.read_parquet(MANIFEST_PATH)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    print("Proxy retrieval:", run_proxy_eval(encoder, val_df, device))
    print("Quadrant probe:", run_quadrant_probe(encoder, val_df, device))
    print("Pattern probe:", run_pattern_probe(encoder, val_df, device, Path(args.buckets)))


if __name__ == "__main__":
    main()

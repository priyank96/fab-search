"""
Sweep re-ranker weight combinations against the validation set.

Uses the finetuned embedding model + cached val page crops to build a
self-retrieval benchmark:
    - Each val patch becomes a query by applying training augmentations.
    - Candidates = all val patches.
    - For each weight combo, score = w_emb*cos + w_color*hsv + w_tex*lbp
      + w_edge*edge + w_dom*dom. Rank, measure Recall@K.

No labels, no Pinecone, no network. Runs on Colab T4/A100.

Usage:
    # weights from local export
    python -m training.sweep_rerank_weights \\
        --weights training/checkpoints/fab_search_vit_b32.pt \\
        --n-queries 200

    # weights from HF
    python -m training.sweep_rerank_weights \\
        --hf-repo knayirp/fab-search-vit-b32 \\
        --n-queries 200
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

from src.reranker import (
    _compare_edge_density,
    _compare_histograms,
    _compare_lbp,
    _dominant_colors,
    _edge_density,
    _hsv_histogram,
    _lbp_histogram,
)
from training.augment import build_eval_transform, build_query_aug, prepare_patch
from training.config import MANIFEST_PATH, MODEL_NAME, PRETRAINED
from training.dataset import _crop_patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sweep_rerank")

_THUMB = 128  # match src.reranker


# ── Model loading ────────────────────────────────────────────
def _load_model(weights_path: str | None, hf_repo: str | None, device: torch.device):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=None if (weights_path or hf_repo) else PRETRAINED)
    if hf_repo:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(repo_id=hf_repo, filename="fab_search_vit_b32.pt")
    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("Loaded weights (missing=%d unexpected=%d)", len(missing), len(unexpected))
    model.visual.to(device).eval()
    return model


@torch.no_grad()
def _encode_image(model, pil_or_np, device, transform):
    """Encode a single image (numpy HWC RGB or PIL) → 512-d normalised."""
    if isinstance(pil_or_np, np.ndarray):
        from PIL import Image
        pil = Image.fromarray(pil_or_np)
    else:
        pil = pil_or_np
    arr = prepare_patch(np.array(pil.convert("RGB")), transform)
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    feat = model.encode_image(t).float()
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze(0).cpu().numpy()


# ── Hand-crafted feature extraction (mirrors src.reranker) ───
def _patch_features(rgb: np.ndarray) -> dict:
    """rgb is HWC uint8. Returns dict of features for a candidate patch."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (_THUMB, _THUMB), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "hsv": _hsv_histogram(bgr),
        "lbp": _lbp_histogram(gray),
        "edge": _edge_density(gray),
        "dom": _dominant_colors(bgr, k=3),
    }


def _dominant_color_sim(dom_a: np.ndarray, dom_b: np.ndarray) -> float:
    """Min-pair LAB distance → similarity in [0, 1] (mirror reranker)."""
    # Greedy min-matching of k×k LAB distances; lower is more similar.
    d = np.linalg.norm(dom_a[:, None, :] - dom_b[None, :, :], axis=-1)  # (k, k)
    # Hungarian-light: take min over each row, average.
    mins = d.min(axis=1)
    avg_d = float(mins.mean())
    # Normalise: LAB distances >50 = essentially unrelated.
    return float(max(0.0, 1.0 - avg_d / 50.0))


# ── Build feature bank for val patches ───────────────────────
def _build_bank(val_df: pd.DataFrame, model, device, transform):
    embs, feats, keep_idx = [], [], []
    for i, row in tqdm(list(val_df.iterrows()), desc="encode val"):
        try:
            patch = _crop_patch(
                row["local_page"],
                int(row["tile_x"]), int(row["tile_y"]),
                int(row["tile_w"]), int(row["tile_h"]),
            )
        except Exception as e:
            log.debug("skip %s: %s", row.get("id"), e)
            continue
        emb = _encode_image(model, patch, device, transform)
        f = _patch_features(np.array(patch))
        embs.append(emb)
        feats.append(f)
        keep_idx.append(i)
    return np.stack(embs), feats, keep_idx


def _score_candidates(
    q_emb, q_feat, bank_embs, bank_feats, weights: dict,
) -> np.ndarray:
    cos = bank_embs @ q_emb  # (N,)
    # Hand-crafted similarities, per-candidate.
    hsv = np.array([_compare_histograms(q_feat["hsv"], f["hsv"]) for f in bank_feats])
    lbp = np.array([_compare_lbp(q_feat["lbp"], f["lbp"]) for f in bank_feats])
    edge = np.array([_compare_edge_density(q_feat["edge"], f["edge"]) for f in bank_feats])
    dom = np.array([_dominant_color_sim(q_feat["dom"], f["dom"]) for f in bank_feats])
    return (
        weights["emb"] * cos
        + weights["color"] * hsv
        + weights["texture"] * lbp
        + weights["edge"] * edge
        + weights["dom"] * dom
    )


def _recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float((ranks < k).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="Local path to exported .pt")
    ap.add_argument("--hf-repo", default=None, help="HF repo id like knayirp/fab-search-vit-b32")
    ap.add_argument("--n-queries", type=int, default=200, help="Number of val patches to use as queries")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    df = pd.read_parquet(MANIFEST_PATH)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    log.info("Val patches: %d", len(val_df))

    model = _load_model(args.weights, args.hf_repo, device)
    eval_tf = build_eval_transform()
    query_aug = build_query_aug()

    log.info("Building feature bank for all val patches ...")
    bank_embs, bank_feats, kept = _build_bank(val_df, model, device, eval_tf)
    bank_df = val_df.iloc[kept].reset_index(drop=True)
    N = len(bank_df)
    log.info("Bank size: %d", N)

    # Sample queries
    rng = np.random.default_rng(args.seed)
    n_q = min(args.n_queries, N)
    q_idx = rng.choice(N, size=n_q, replace=False)

    # Generate augmented queries (one heavy aug per source patch)
    log.info("Generating %d augmented queries ...", n_q)
    queries = []  # list of (true_idx, q_emb, q_feat)
    for qi in tqdm(q_idx, desc="queries"):
        row = bank_df.iloc[qi]
        try:
            patch = _crop_patch(
                row["local_page"], int(row["tile_x"]), int(row["tile_y"]),
                int(row["tile_w"]), int(row["tile_h"]),
            )
        except Exception:
            continue
        rgb = np.array(patch)
        aug_rgb = query_aug(image=rgb)["image"]  # albumentations dict-style
        q_emb = _encode_image(model, aug_rgb, device, eval_tf)
        q_feat = _patch_features(aug_rgb if aug_rgb.dtype == np.uint8 else (aug_rgb * 255).astype(np.uint8))
        queries.append((int(qi), q_emb, q_feat))
    log.info("Built %d queries", len(queries))

    # Weight combinations to sweep
    combos = [
        # name,                 (emb, color, texture, edge, dom)
        ("current (0.35/0.20/0.20/0.10/0.15)", (0.35, 0.20, 0.20, 0.10, 0.15)),
        ("embedding-heavy (0.50/0.20/0.10/0.05/0.15)", (0.50, 0.20, 0.10, 0.05, 0.15)),
        ("embedding-very-heavy (0.60/0.15/0.10/0.05/0.10)", (0.60, 0.15, 0.10, 0.05, 0.10)),
        ("embedding-dominant (0.70/0.15/0.05/0.00/0.10)", (0.70, 0.15, 0.05, 0.00, 0.10)),
        ("embedding-only (1.0)", (1.00, 0.00, 0.00, 0.00, 0.00)),
        ("colour-only (0/0.5/0/0/0.5)", (0.00, 0.50, 0.00, 0.00, 0.50)),
        ("balanced (0.40/0.20/0.15/0.10/0.15)", (0.40, 0.20, 0.15, 0.10, 0.15)),
        ("reduce texture (0.45/0.20/0.05/0.10/0.20)", (0.45, 0.20, 0.05, 0.10, 0.20)),
    ]

    results = []
    for name, (we, wc, wt, wed, wd) in combos:
        w = {"emb": we, "color": wc, "texture": wt, "edge": wed, "dom": wd}
        ranks = []
        for true_idx, q_emb, q_feat in queries:
            scores = _score_candidates(q_emb, q_feat, bank_embs, bank_feats, w)
            order = np.argsort(-scores)
            rank = int(np.where(order == true_idx)[0][0])
            ranks.append(rank)
        ranks = np.asarray(ranks)
        r1 = _recall_at_k(ranks, 1)
        r5 = _recall_at_k(ranks, 5)
        r10 = _recall_at_k(ranks, 10)
        mrr = float((1.0 / (ranks + 1)).mean())
        log.info("%-55s  R@1=%.3f  R@5=%.3f  R@10=%.3f  MRR=%.3f", name, r1, r5, r10, mrr)
        results.append((name, r1, r5, r10, mrr))

    print("\n=== Summary (sorted by Recall@5) ===")
    results.sort(key=lambda r: -r[2])
    for name, r1, r5, r10, mrr in results:
        print(f"  R@5={r5:.3f}  R@1={r1:.3f}  R@10={r10:.3f}  MRR={mrr:.3f}  {name}")


if __name__ == "__main__":
    main()

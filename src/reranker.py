from __future__ import annotations

"""
fab-search — Multi-feature re-ranking.

After initial embedding-based retrieval from Pinecone, re-rank candidates
using additional patch-level visual signals computed on tile crops:

1. Color histogram: HSV distribution similarity.
2. Texture (LBP): Local Binary Pattern histogram captures weave/pattern
   structure — critical for distinguishing fabrics with similar colour
   but different textures.
3. Edge density: ratio of Canny edge pixels — separates busy/detailed
   patterns from plain/solid fabrics.
4. Dominant colour match: k-means top-3 colours compared via earth-mover
   style min-distance — more robust than full histogram when a fabric
   has only a few key colours.
"""

import io
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from PIL import Image

from src.config import (
    RERANK_FETCH_K,
    RERANK_PATCH_CANDIDATES,
    W_EMBEDDING,
    W_COLOR,
    W_TEXTURE,
    W_EDGE,
    W_DOMINANT_COLOR,
)

log = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 8  # seconds per image
_THUMB = 128  # downscale crops for feature extraction

# ── Feature helpers ──────────────────────────────────────────

# -- Color histogram (HSV) --
_HSV_BINS = [12, 10, 10]
_HSV_RANGES = [0, 180, 0, 256, 0, 256]


def _hsv_histogram(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, _HSV_BINS, _HSV_RANGES)
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _compare_histograms(h1: np.ndarray, h2: np.ndarray) -> float:
    score = cv2.compareHist(h1.reshape(-1, 1), h2.reshape(-1, 1), cv2.HISTCMP_CORREL)
    return float(max(0.0, score))


# -- Texture (LBP) --
def _lbp_histogram(gray: np.ndarray, n_bins: int = 26) -> np.ndarray:
    """Compute a simple rotation-invariant LBP histogram."""
    h, w = gray.shape
    # Pad to avoid boundary checks
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = padded[1:-1, 1:-1].astype(np.int16)
    # 8 neighbours
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].astype(np.int16)
        lbp |= ((neighbour >= center).astype(np.uint8) << bit)
    # Uniform mapping: count bit transitions; patterns with <=2 transitions
    # get their own bin, all others map to a single "non-uniform" bin.
    hist = np.zeros(n_bins, dtype=np.float32)
    flat = lbp.ravel()
    for val in flat:
        bitstr = np.binary_repr(val, width=8)
        transitions = sum(bitstr[i] != bitstr[i + 1] for i in range(7)) + (bitstr[7] != bitstr[0])
        if transitions <= 2:
            n_ones = bin(val).count("1")
            # bin index = number of 1-bits (0..8) → bins 0-8
            # But we shift uniform patterns by their rotation-invariant id
            hist[n_ones] += 1
        else:
            hist[n_bins - 1] += 1  # non-uniform bin
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _compare_lbp(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared distance → similarity in [0, 1]."""
    chi2 = 0.5 * np.sum(np.where(h1 + h2 > 0, (h1 - h2) ** 2 / (h1 + h2), 0.0))
    return float(1.0 / (1.0 + chi2))


# -- Edge density --
def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are edges (Canny)."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / max(edges.size, 1))


def _compare_edge_density(d1: float, d2: float) -> float:
    """Similarity based on edge density ratio. 1.0 = identical density."""
    if d1 == 0.0 and d2 == 0.0:
        return 1.0
    ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0
    return ratio


# -- Dominant colour (k-means) --
def _dominant_colors(bgr: np.ndarray, k: int = 3) -> np.ndarray:
    """Extract k dominant colours via k-means. Returns (k, 3) float32 in LAB."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # Subsample for speed
    if lab.shape[0] > 2000:
        indices = np.random.default_rng(42).choice(lab.shape[0], 2000, replace=False)
        lab = lab[indices]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers  # (k, 3)


def _compare_dominant_colors(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Min-distance matching between two sets of dominant colours (LAB space).
    Returns similarity in [0, 1].
    """
    # For each colour in c1, find nearest in c2, and vice versa
    dists_1to2 = np.min(np.linalg.norm(c1[:, None] - c2[None, :], axis=2), axis=1)
    dists_2to1 = np.min(np.linalg.norm(c2[:, None] - c1[None, :], axis=2), axis=1)
    avg_dist = (dists_1to2.mean() + dists_2to1.mean()) / 2.0
    # LAB distance: ~0 = identical, ~100+ = very different
    return float(max(0.0, 1.0 - avg_dist / 100.0))


# ── Image fetching ───────────────────────────────────────────

def _download_tile_image(
    page_image_url: str,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray | None:
    """Download page image, crop to tile, return as BGR numpy array."""
    try:
        resp = requests.get(page_image_url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        cropped = img.crop((tile_x, tile_y, tile_x + tile_w, tile_y + tile_h))
        # Release the full-resolution page buffer immediately — it's the
        # largest allocation here and we only need the small crop downstream.
        img.close()
        cropped.thumbnail((_THUMB, _THUMB), Image.LANCZOS)
        return cv2.cvtColor(np.asarray(cropped), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("Failed to download tile from %s: %s", page_image_url, e)
        return None


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR numpy array, downscaled."""
    img = pil_image.copy()
    img.thumbnail((_THUMB, _THUMB), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


# ── Precompute all features for one image ────────────────────

def _extract_features(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "hsv_hist": _hsv_histogram(bgr),
        "lbp_hist": _lbp_histogram(gray),
        "edge_density": _edge_density(gray),
        "dominant_colors": _dominant_colors(bgr),
    }


# ── Main re-ranker ───────────────────────────────────────────


def rerank(
    raw_matches: list[dict],
    query_image: Image.Image,
    top_k: int = 5,
) -> list[dict]:
    """
    Re-rank raw Pinecone matches using patch-level visual features.

    Args:
        raw_matches: Flat list of match dicts from query_raw().
        query_image: Original query PIL image.
        top_k: Number of final results to return.

    Returns:
        Ranked list of result dicts with score breakdown fields.
    """
    if not raw_matches:
        return []

    # ── Phase 1: Group by page, pick best tile per page ──────

    page_groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in raw_matches:
        key = (m["pdf_name"], m["page_number"])
        page_groups[key].append(m)

    candidates: list[dict] = []
    for page_key, tiles in page_groups.items():
        best_tile = max(tiles, key=lambda t: t["score"])
        candidates.append({
            "page_key": page_key,
            "best_tile": best_tile,
            "max_score": float(best_tile["score"]),
            "color_sim": 0.5,
            "texture_sim": 0.5,
            "edge_sim": 0.5,
            "dominant_color_sim": 0.5,
        })

    # Pre-sort by embedding score so patch comparison targets the best
    candidates.sort(key=lambda c: c["max_score"], reverse=True)

    # ── Fast-path: embedding-only ────────────────────────────
    # When the embedding fully owns the ranking, skip tile downloads and
    # feature extraction entirely (saves ~1-2s per query).
    if W_EMBEDDING >= 0.9999 and (W_COLOR + W_TEXTURE + W_EDGE + W_DOMINANT_COLOR) <= 1e-6:
        log.info("Embedding-only mode (W_EMBEDDING=%.2f); skipping phase 2", W_EMBEDDING)
        results: list[dict] = []
        for c in candidates[:top_k]:
            bt = c["best_tile"]
            results.append({
                "rank": len(results) + 1,
                "score": round(c["max_score"], 4),
                "embedding_score": round(c["max_score"], 4),
                "color_score": 0.0,
                "texture_score": 0.0,
                "edge_score": 0.0,
                "dominant_color_score": 0.0,
                **{k: bt[k] for k in ("pdf_name", "page_number", "tile_x", "tile_y", "tile_w", "tile_h", "page_image")},
            })
        return results

    # ── Phase 2: Patch-level feature comparison ──────────────

    # Precompute query features once
    try:
        query_bgr = _pil_to_bgr(query_image)
        query_feats = _extract_features(query_bgr)
    except Exception:
        log.warning("Could not extract query features; returning embedding-only ranking")
        # Fall back to embedding-only
        results: list[dict] = []
        for c in candidates[:top_k]:
            bt = c["best_tile"]
            results.append({
                "rank": len(results) + 1,
                "score": round(c["max_score"], 4),
                "embedding_score": round(c["max_score"], 4),
                "color_score": 0.0,
                "texture_score": 0.0,
                "edge_score": 0.0,
                "dominant_color_score": 0.0,
                **{k: bt[k] for k in ("pdf_name", "page_number", "tile_x", "tile_y", "tile_w", "tile_h", "page_image")},
            })
        return results

    n_patch = min(RERANK_PATCH_CANDIDATES, len(candidates))
    top_cands = candidates[:n_patch]

    # Download tile images in parallel
    def _fetch(idx: int) -> tuple[int, np.ndarray | None]:
        bt = top_cands[idx]["best_tile"]
        return idx, _download_tile_image(
            bt["page_image"], bt["tile_x"], bt["tile_y"],
            bt["tile_w"], bt["tile_h"],
        )

    tile_images: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_fetch, i): i for i in range(n_patch)}
        for future in as_completed(futures):
            idx, bgr = future.result()
            if bgr is not None:
                tile_images[idx] = bgr

    # Compute features and compare
    for idx, bgr in tile_images.items():
        try:
            tile_feats = _extract_features(bgr)
            top_cands[idx]["color_sim"] = _compare_histograms(
                query_feats["hsv_hist"], tile_feats["hsv_hist"]
            )
            top_cands[idx]["texture_sim"] = _compare_lbp(
                query_feats["lbp_hist"], tile_feats["lbp_hist"]
            )
            top_cands[idx]["edge_sim"] = _compare_edge_density(
                query_feats["edge_density"], tile_feats["edge_density"]
            )
            top_cands[idx]["dominant_color_sim"] = _compare_dominant_colors(
                query_feats["dominant_colors"], tile_feats["dominant_colors"]
            )
        except Exception:
            log.debug("Feature extraction failed for candidate %d", idx)

    # ── Phase 3: Weighted composite score ────────────────────

    for c in candidates:
        c["composite_score"] = (
            W_EMBEDDING * c["max_score"]
            + W_COLOR * c["color_sim"]
            + W_TEXTURE * c["texture_sim"]
            + W_EDGE * c["edge_sim"]
            + W_DOMINANT_COLOR * c["dominant_color_sim"]
        )

    candidates.sort(key=lambda c: c["composite_score"], reverse=True)

    # ── Build output ─────────────────────────────────────────

    results: list[dict] = []
    for c in candidates[:top_k]:
        bt = c["best_tile"]
        results.append({
            "rank": len(results) + 1,
            "score": round(c["composite_score"], 4),
            "embedding_score": round(c["max_score"], 4),
            "color_score": round(c["color_sim"], 4),
            "texture_score": round(c["texture_sim"], 4),
            "edge_score": round(c["edge_sim"], 4),
            "dominant_color_score": round(c["dominant_color_sim"], 4),
            "pdf_name": bt["pdf_name"],
            "page_number": bt["page_number"],
            "tile_x": bt["tile_x"],
            "tile_y": bt["tile_y"],
            "tile_w": bt["tile_w"],
            "tile_h": bt["tile_h"],
            "page_image": bt["page_image"],
        })

    return results

"""
Build the training dataset manifest from the live Pinecone index.

Steps:
1. Enumerate all vectors in Pinecone (paginated via index.list()).
2. Fetch metadata for each vector.
3. Filter out full-page rows (bbox covers the whole page).
4. Cache each unique page_image JPEG locally.
5. Write a parquet manifest.

Run:
    python -m training.build_dataset

Env:
    PINECONE_API_KEY, PINECONE_INDEX_NAME — same as production.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from PIL import Image
from pinecone import Pinecone
from tqdm import tqdm

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from training.config import (
    DEFAULT_NUM_VAL_CATALOGS,
    MANIFEST_PATH,
    PAGE_CACHE_DIR,
    VAL_CATALOGS,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FETCH_BATCH = 100  # Pinecone fetch batch size


def _iter_vector_ids(index) -> Iterable[str]:
    """Stream vector IDs from Pinecone using the list endpoint.

    Falls back to a zero-vector query for tiers that don't expose list().
    """
    try:
        for page in index.list(limit=100):
            yield from page
        return
    except Exception as exc:  # pragma: no cover — fallback path
        log.warning("index.list() failed (%s); falling back to query()", exc)

    # Fallback: zero-vector top_k query (works up to ~10k vectors).
    from src.config import EMBEDDING_DIM
    dummy = [0.0] * EMBEDDING_DIM
    res = index.query(vector=dummy, top_k=10000, include_metadata=False)
    for m in res.matches:
        yield m.id


def _as_plain(obj):
    """Normalise Pinecone response/vector objects to plain Python (dict/list)."""
    if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Last resort: pull common attrs.
    out = {}
    for k in ("id", "metadata", "values", "vectors", "namespace"):
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out


def _fetch_metadata(index, ids: list[str], _debug: list[bool] = [True]) -> list[dict]:
    """Fetch metadata for a batch of vector IDs.

    The Pinecone client has shifted shapes a few times. We normalise the
    whole response via to_dict()/model_dump() so we always operate on
    plain dicts/lists.
    """
    rows: list[dict] = []
    for i in range(0, len(ids), FETCH_BATCH):
        batch = ids[i : i + FETCH_BATCH]
        resp = _as_plain(index.fetch(ids=batch))

        vectors = None
        if isinstance(resp, dict):
            vectors = resp.get("vectors")
        if vectors is None and hasattr(resp, "vectors"):
            vectors = _as_plain(resp.vectors)

        if _debug[0]:
            log.info("fetch resp keys=%s vectors type=%s len=%s",
                     list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__,
                     type(vectors).__name__,
                     (len(vectors) if hasattr(vectors, "__len__") else "?"))
            if vectors:
                if isinstance(vectors, dict):
                    first_id = next(iter(vectors))
                    first_val = _as_plain(vectors[first_id])
                else:
                    first_val = _as_plain(vectors[0])
                    first_id = first_val.get("id") if isinstance(first_val, dict) else "?"
                log.info("first vector id=%s payload=%r", first_id, first_val)
            _debug[0] = False

        if not vectors:
            continue

        # vectors may be a dict {id: vec} or a list[vec].
        iterable = vectors.items() if isinstance(vectors, dict) else (
            ((_as_plain(v).get("id"), v) for v in vectors)
        )
        for vid, vec in iterable:
            vec = _as_plain(vec)
            if not isinstance(vec, dict):
                continue
            meta = vec.get("metadata")
            if not meta:
                continue
            rows.append({"id": vid or vec.get("id"), **meta})
    return rows


def _is_full_page(row: dict, page_size: tuple[int, int] | None = None) -> bool:
    """Heuristic: full-page vectors have tile_x=tile_y=0 and tile_w/h = page size.

    We don't know page size up front, so we use a simpler proxy:
    tile_x == 0 and tile_y == 0 and tile_w >= 1500 and tile_h >= 1500
    (200 DPI A4 = ~1654x2339 px).
    """
    return (
        int(row.get("tile_x", -1)) == 0
        and int(row.get("tile_y", -1)) == 0
        and int(row.get("tile_w", 0)) >= 1500
        and int(row.get("tile_h", 0)) >= 1500
    )


def _safe_filename(url: str) -> str:
    # page_image URLs look like "{R2_PUBLIC_URL}/pages/{pdf}/{stem}_page_NNNN.jpg".
    # Use the path tail, replacing slashes with __.
    tail = url.split("//", 1)[-1].split("/", 1)[-1]
    return tail.replace("/", "__")


def _download_pages(urls: set[str]) -> dict[str, Path]:
    """Download each unique URL once into PAGE_CACHE_DIR. Returns url -> local path."""
    mapping: dict[str, Path] = {}
    for url in tqdm(sorted(urls), desc="cache pages"):
        local = PAGE_CACHE_DIR / _safe_filename(url)
        mapping[url] = local
        if local.exists() and local.stat().st_size > 0:
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            local.write_bytes(r.content)
            # Verify it decodes
            with Image.open(local) as im:
                im.verify()
        except Exception as exc:
            log.warning("failed to download %s: %s", url, exc)
            if local.exists():
                local.unlink(missing_ok=True)
    return mapping


def _pick_val_catalogs(df: pd.DataFrame) -> list[str]:
    if VAL_CATALOGS:
        return VAL_CATALOGS
    counts = df.groupby("pdf_name").size().sort_values()
    # Smallest catalogs, but skip any with <10 patches (too few to evaluate on).
    candidates = [n for n, c in counts.items() if c >= 10]
    return candidates[:DEFAULT_NUM_VAL_CATALOGS]


def main() -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    log.info("Enumerating vector IDs from %s...", PINECONE_INDEX_NAME)
    ids = list(_iter_vector_ids(index))
    log.info("Found %d vectors", len(ids))
    if not ids:
        raise SystemExit("No vectors in Pinecone index. Run src.index_cli first.")

    log.info("Fetching metadata...")
    rows: list[dict] = []
    for i in tqdm(range(0, len(ids), FETCH_BATCH), desc="fetch meta"):
        rows.extend(_fetch_metadata(index, ids[i : i + FETCH_BATCH]))

    df = pd.DataFrame(rows)
    log.info("Raw rows: %d", len(df))

    # Drop full-page entries (no useful patch).
    full_mask = df.apply(_is_full_page, axis=1)
    log.info("Dropping %d full-page rows", int(full_mask.sum()))
    df = df.loc[~full_mask].reset_index(drop=True)

    # Coerce types.
    for col in ("page_number", "tile_x", "tile_y", "tile_w", "tile_h"):
        df[col] = df[col].astype(int)

    # Cache page JPEGs.
    urls = set(df["page_image"].dropna().unique())
    log.info("Caching %d unique page images...", len(urls))
    url_to_path = _download_pages(urls)
    df["local_page"] = df["page_image"].map(lambda u: str(url_to_path.get(u, "")))
    missing = (df["local_page"] == "").sum()
    if missing:
        log.warning("%d rows have no local page image; dropping", missing)
        df = df.loc[df["local_page"] != ""].reset_index(drop=True)

    # Train/val split by pdf_name.
    val_catalogs = _pick_val_catalogs(df)
    log.info("Val catalogs (%d): %s", len(val_catalogs), val_catalogs)
    df["split"] = df["pdf_name"].map(lambda n: "val" if n in val_catalogs else "train")

    # The single-pattern flag is filled in by a separate pass that needs the
    # pretrained CLIP model loaded; we leave it null here. See
    # training.flag_single_pattern for that step.
    df["single_pattern"] = pd.NA

    # Persist.
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST_PATH, index=False)
    log.info("Wrote %d rows -> %s", len(df), MANIFEST_PATH)
    log.info("Splits: %s", df["split"].value_counts().to_dict())
    log.info("Catalogs: %d unique", df["pdf_name"].nunique())


if __name__ == "__main__":
    main()

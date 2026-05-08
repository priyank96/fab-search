"""
Build the training dataset manifest from the live Pinecone index.

Approach (mirrors the proven pattern in src/pinecone_store.py — uses
.query() + include_metadata=True throughout, since .fetch()/.list()
behave inconsistently across Pinecone client versions and tiers):

1. Discover catalog names by running one zero-vector query and reading
   the unique `pdf_name` values from match metadata.
2. For each catalog, run a filtered zero-vector query (top_k=10000) to
   pull every vector + metadata for that catalog.
3. Drop full-page rows (bbox covers the whole page).
4. Cache each unique page_image JPEG locally.
5. Write parquet manifest.

Run:
    python -m training.build_dataset

Env:
    PINECONE_API_KEY, PINECONE_INDEX_NAME — same as production.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from PIL import Image
from pinecone import Pinecone
from tqdm import tqdm

from src.config import EMBEDDING_DIM, PINECONE_API_KEY, PINECONE_INDEX_NAME
from training.config import (
    DEFAULT_NUM_VAL_CATALOGS,
    MANIFEST_PATH,
    PAGE_CACHE_DIR,
    VAL_CATALOGS,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

QUERY_TOP_K = 10000  # Pinecone hard cap


def _zero_vec() -> list[float]:
    return [0.0] * EMBEDDING_DIM


def _list_catalogs(index) -> list[str]:
    """Discover catalog names via one broad zero-vector query."""
    resp = index.query(
        vector=_zero_vec(),
        top_k=QUERY_TOP_K,
        include_metadata=True,
    )
    names: set[str] = set()
    for m in resp.matches:
        meta = m.metadata or {}
        n = meta.get("pdf_name")
        if n:
            names.add(str(n))
    return sorted(names)


def _fetch_catalog_rows(index, pdf_name: str) -> list[dict]:
    """Pull every vector for one catalog. Returns flat list of metadata dicts."""
    resp = index.query(
        vector=_zero_vec(),
        top_k=QUERY_TOP_K,
        include_metadata=True,
        filter={"pdf_name": {"$eq": pdf_name}},
    )
    rows: list[dict] = []
    for m in resp.matches:
        meta = m.metadata or {}
        if not meta:
            continue
        rows.append({"id": str(m.id), **dict(meta)})
    return rows


def _is_full_page(row: dict) -> bool:
    """Heuristic: tile_x=tile_y=0 and tile_w/h >= 1500 (≈ A4 @ 200 DPI)."""
    return (
        int(row.get("tile_x", -1)) == 0
        and int(row.get("tile_y", -1)) == 0
        and int(row.get("tile_w", 0)) >= 1500
        and int(row.get("tile_h", 0)) >= 1500
    )


def _safe_filename(url: str) -> str:
    tail = url.split("//", 1)[-1].split("/", 1)[-1]
    return tail.replace("/", "__")


def _download_pages(urls: set[str]) -> dict[str, str]:
    """Download each unique URL once. Returns url -> local path string ('' on failure)."""
    mapping: dict[str, str] = {}
    for url in tqdm(sorted(urls), desc="cache pages"):
        local = PAGE_CACHE_DIR / _safe_filename(url)
        mapping[url] = str(local)
        if local.exists() and local.stat().st_size > 0:
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            local.write_bytes(r.content)
            with Image.open(local) as im:
                im.verify()
        except Exception as exc:
            log.warning("failed to download %s: %s", url, exc)
            if local.exists():
                local.unlink(missing_ok=True)
            mapping[url] = ""
    return mapping


def _pick_val_catalogs(df: pd.DataFrame) -> list[str]:
    if VAL_CATALOGS:
        return VAL_CATALOGS
    counts = df.groupby("pdf_name").size().sort_values()
    candidates = [n for n, c in counts.items() if c >= 10]
    return candidates[:DEFAULT_NUM_VAL_CATALOGS]


def main() -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()
    log.info("Index stats: %s", stats)

    log.info("Discovering catalogs...")
    catalogs = _list_catalogs(index)
    log.info("Found %d catalog(s): %s", len(catalogs), catalogs)
    if not catalogs:
        raise SystemExit("No catalogs found. Run src.index_cli first.")

    all_rows: list[dict] = []
    for name in tqdm(catalogs, desc="fetch catalogs"):
        rows = _fetch_catalog_rows(index, name)
        log.info("  %s: %d rows", name, len(rows))
        all_rows.extend(rows)
    log.info("Total raw rows: %d", len(all_rows))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("Empty result set; aborting.")

    full_mask = df.apply(_is_full_page, axis=1)
    log.info("Dropping %d full-page rows", int(full_mask.sum()))
    df = df.loc[~full_mask].reset_index(drop=True)

    for col in ("page_number", "tile_x", "tile_y", "tile_w", "tile_h"):
        df[col] = df[col].astype(float).astype(int)

    urls = set(df["page_image"].dropna().unique())
    log.info("Caching %d unique page images...", len(urls))
    url_to_path = _download_pages(urls)
    df["local_page"] = df["page_image"].map(lambda u: url_to_path.get(u, ""))
    missing = (df["local_page"] == "").sum()
    if missing:
        log.warning("%d rows have no local page image; dropping", missing)
        df = df.loc[df["local_page"] != ""].reset_index(drop=True)

    val_catalogs = _pick_val_catalogs(df)
    log.info("Val catalogs (%d): %s", len(val_catalogs), val_catalogs)
    df["split"] = df["pdf_name"].map(lambda n: "val" if n in val_catalogs else "train")

    df["single_pattern"] = pd.NA  # filled in by training.flag_single_pattern

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST_PATH, index=False)
    log.info("Wrote %d rows -> %s", len(df), MANIFEST_PATH)
    log.info("Splits: %s", df["split"].value_counts().to_dict())
    log.info("Catalogs: %d unique", df["pdf_name"].nunique())


if __name__ == "__main__":
    main()

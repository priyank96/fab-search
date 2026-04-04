from __future__ import annotations

"""
fab-search — Pinecone vector store with metadata.

Replaces FAISS + SQLite from FabricSearch. All tile metadata is stored
directly on each Pinecone vector, eliminating the need for a sidecar
database.

Pinecone metadata fields per vector:
    pdf_name     (str)  — source PDF filename
    page_number  (int)  — 1-based page number
    tile_x       (int)  — top-left x of the tile/region on the page
    tile_y       (int)  — top-left y
    tile_w       (int)  — width in pixels
    tile_h       (int)  — height in pixels
    page_image   (str)  — URL or path to the rendered page image
"""

import uuid
import logging
import numpy as np
from pinecone import Pinecone

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIM

log = logging.getLogger(__name__)


def _get_pinecone_index():
    """Connect to the Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


class PineconeStore:
    """
    Wrapper around a Pinecone index for adding, querying, and managing
    catalog vectors with tile metadata.

    Usage:
        store = PineconeStore()
        store.add(embedding, pdf_name="catalog.pdf", page_number=1, ...)
        results = store.query(query_embedding, top_k=5)
        store.remove_catalog("catalog.pdf")
    """

    def __init__(self):
        self.index = _get_pinecone_index()
        self._pending: list[dict] = []

    # ── Core operations ───────────────────────────────────────

    def add(
        self,
        embedding: np.ndarray,
        pdf_name: str,
        page_number: int,
        tile_x: int,
        tile_y: int,
        tile_w: int,
        tile_h: int,
        page_image: str,
    ) -> str:
        """
        Stage a vector + metadata for upsert.

        Call flush() to persist to Pinecone.
        Returns the generated vector ID.
        """
        vec_id = str(uuid.uuid4())
        self._pending.append({
            "id": vec_id,
            "values": embedding.tolist(),
            "metadata": {
                "pdf_name": pdf_name,
                "page_number": page_number,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "page_image": page_image,
            },
        })

        # Auto-flush in batches of 100 (Pinecone recommended batch size)
        if len(self._pending) >= 100:
            self.flush()

        return vec_id

    def flush(self) -> int:
        """
        Upsert all pending vectors to Pinecone.

        Returns the number of vectors upserted.
        """
        if not self._pending:
            return 0

        count = len(self._pending)
        # Pinecone upsert accepts list of (id, values, metadata) tuples
        self.index.upsert(vectors=self._pending)
        self._pending.clear()
        return count

    def close(self) -> None:
        """Flush any remaining vectors."""
        self.flush()

    # ── Query ─────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        catalog_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Search for similar vectors, deduplicating by page.

        Args:
            query_embedding: 512-dim CLIP embedding.
            top_k: Number of unique pages to return.
            catalog_filter: If provided, restrict results to these pdf_names.

        Returns a list of result dicts with keys:
            rank, score, pdf_name, page_number,
            tile_x, tile_y, tile_w, tile_h, page_image
        """
        # Over-fetch to allow page deduplication
        fetch_k = min(top_k * 20, 1000)

        query_kwargs = dict(
            vector=query_embedding.tolist(),
            top_k=fetch_k,
            include_metadata=True,
        )
        if catalog_filter:
            query_kwargs["filter"] = {"pdf_name": {"$in": catalog_filter}}

        response = self.index.query(**query_kwargs)

        seen_pages: set[tuple[str, int]] = set()
        results: list[dict] = []

        for match in response.matches:
            if len(results) >= top_k:
                break

            meta = match.metadata
            page_key = (meta["pdf_name"], meta["page_number"])
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)

            results.append({
                "rank": len(results) + 1,
                "score": float(match.score),
                "pdf_name": meta["pdf_name"],
                "page_number": meta["page_number"],
                "tile_x": meta["tile_x"],
                "tile_y": meta["tile_y"],
                "tile_w": meta["tile_w"],
                "tile_h": meta["tile_h"],
                "page_image": meta["page_image"],
            })

        return results

    def query_raw(
        self,
        query_embedding: np.ndarray,
        fetch_k: int = 200,
        catalog_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Fetch the top *fetch_k* matches without page deduplication.

        Used by the reranker, which handles grouping and scoring itself.

        Returns a flat list of dicts with keys:
            score, pdf_name, page_number,
            tile_x, tile_y, tile_w, tile_h, page_image
        """
        query_kwargs = dict(
            vector=query_embedding.tolist(),
            top_k=min(fetch_k, 1000),
            include_metadata=True,
        )
        if catalog_filter:
            query_kwargs["filter"] = {"pdf_name": {"$in": catalog_filter}}

        response = self.index.query(**query_kwargs)

        results: list[dict] = []
        for match in response.matches:
            meta = match.metadata
            results.append({
                "score": float(match.score),
                "pdf_name": meta["pdf_name"],
                "page_number": meta["page_number"],
                "tile_x": meta["tile_x"],
                "tile_y": meta["tile_y"],
                "tile_w": meta["tile_w"],
                "tile_h": meta["tile_h"],
                "page_image": meta["page_image"],
            })

        return results

    # ── Catalog management ────────────────────────────────────

    def remove_catalog(self, pdf_name: str) -> int:
        """
        Remove all vectors for a given catalog.

        Uses metadata filtering to find vectors, then deletes by ID.
        Returns the number of vectors removed.
        """
        log.info("remove_catalog: deleting vectors with pdf_name=%r", pdf_name)
        total_removed = 0

        # Pinecone delete with metadata filter
        # This works on serverless and starter indexes
        self.index.delete(filter={"pdf_name": {"$eq": pdf_name}})

        # Note: Pinecone delete-by-filter doesn't return a count on all tiers.
        # We return -1 to indicate "deleted but count unknown" if needed,
        # but practically the delete succeeds.
        return total_removed

    def list_catalogs(self) -> list[dict]:
        """
        List all indexed catalogs with page and vector counts.

        Uses a dummy query to sample vectors and extract unique pdf_names.
        For accurate counts on large indexes, consider maintaining a
        separate metadata store or using Pinecone's list endpoint.

        Returns list of dicts with keys: pdf_name, num_pages, num_vectors.
        """
        # Query with a zero vector to get a broad sample
        dummy_vec = [0.0] * EMBEDDING_DIM
        results = self.index.query(
            vector=dummy_vec,
            top_k=10000,
            include_metadata=True,
        )

        catalog_stats: dict[str, dict[str, set | int]] = {}

        for match in results.matches:
            meta = match.metadata
            name = meta["pdf_name"]
            if name not in catalog_stats:
                catalog_stats[name] = {"pages": set(), "vectors": 0}
            catalog_stats[name]["pages"].add(meta["page_number"])
            catalog_stats[name]["vectors"] += 1

        log.info("list_catalogs: found %d catalog(s): %s",
                 len(catalog_stats),
                 [name for name in sorted(catalog_stats.keys())])

        return [
            {
                "pdf_name": name,
                "num_pages": len(stats["pages"]),
                "num_vectors": stats["vectors"],
            }
            for name, stats in sorted(catalog_stats.items())
        ]

    def get_catalog_blob_urls(self, pdf_name: str) -> list[str]:
        """
        Get unique page_image blob URLs for a catalog.

        Queries Pinecone for all vectors of the given catalog and
        collects the distinct page_image URLs (one per page).
        """
        log.info("get_catalog_blob_urls: querying with pdf_name=%r", pdf_name)
        dummy_vec = [0.0] * EMBEDDING_DIM
        results = self.index.query(
            vector=dummy_vec,
            top_k=10000,
            include_metadata=True,
            filter={"pdf_name": {"$eq": pdf_name}},
        )
        urls: set[str] = set()
        for match in results.matches:
            url = match.metadata.get("page_image")
            if url:
                urls.add(url)
        log.info("get_catalog_blob_urls: found %d unique blob URLs", len(urls))
        return list(urls)

    def has_catalog(self, pdf_name: str) -> bool:
        """Check whether a catalog exists in the index."""
        log.info("has_catalog: checking pdf_name=%r", pdf_name)
        dummy_vec = [1.0] * EMBEDDING_DIM
        results = self.index.query(
            vector=dummy_vec,
            top_k=1,
            include_metadata=True,
            filter={"pdf_name": {"$eq": pdf_name}},
        )
        found = len(results.matches) > 0
        log.info("has_catalog: pdf_name=%r -> %s", pdf_name, found)
        return found

    # ── Page browsing ─────────────────────────────────────────

    def get_catalog_pages(self, pdf_name: str) -> list[dict]:
        """
        Return sorted list of pages for a catalog.

        Each entry: {"page_number": int, "page_image": str}
        """
        log.info("get_catalog_pages: pdf_name=%r", pdf_name)
        dummy_vec = [0.0] * EMBEDDING_DIM
        results = self.index.query(
            vector=dummy_vec,
            top_k=10000,
            include_metadata=True,
            filter={"pdf_name": {"$eq": pdf_name}},
        )
        pages: dict[int, str] = {}
        for match in results.matches:
            meta = match.metadata
            pn = meta["page_number"]
            if pn not in pages:
                pages[pn] = meta.get("page_image", "")
        return [
            {"page_number": pn, "page_image": pages[pn]}
            for pn in sorted(pages)
        ]

    def get_page_tiles(self, pdf_name: str, page_number: int) -> dict:
        """
        Return all tile bounding boxes for a specific page.

        Returns:
            {"page_image": str, "tiles": [{"tile_x", "tile_y", "tile_w", "tile_h"}, ...]}
        """
        log.info("get_page_tiles: pdf_name=%r page=%d", pdf_name, page_number)
        dummy_vec = [0.0] * EMBEDDING_DIM
        results = self.index.query(
            vector=dummy_vec,
            top_k=10000,
            include_metadata=True,
            filter={
                "pdf_name": {"$eq": pdf_name},
                "page_number": {"$eq": page_number},
            },
        )
        page_image = ""
        tiles: list[dict] = []
        for match in results.matches:
            meta = match.metadata
            if not page_image:
                page_image = meta.get("page_image", "")
            tiles.append({
                "tile_x": meta["tile_x"],
                "tile_y": meta["tile_y"],
                "tile_w": meta["tile_w"],
                "tile_h": meta["tile_h"],
            })
        return {"page_image": page_image, "tiles": tiles}

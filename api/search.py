from __future__ import annotations

"""
fab-search — Search API endpoint.
"""

import io
from fastapi import APIRouter, UploadFile, File, Query
from PIL import Image

from src.config import TOP_K, RERANK_FETCH_K
from src.embedding_client import encode_image
from src.pinecone_store import PineconeStore
from src.reranker import rerank

router = APIRouter()


@router.post("/api/search")
async def search(
    file: UploadFile = File(..., description="Query image (jpg/png/webp)"),
    top_k: int = Query(default=TOP_K, ge=1, le=20, description="Number of results"),
    catalogs: str = Query(default="", description="Comma-separated catalog names to search"),
):
    """
    Search the catalog index with an uploaded image.

    Returns ranked results with page image URLs and highlight coordinates.
    """
    contents = await file.read()
    query_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Get embedding from HF Space
    query_embedding = encode_image(query_image)

    # Parse optional catalog filter
    catalog_filter = [c.strip() for c in catalogs.split(",") if c.strip()] or None

    # Fetch raw candidates from Pinecone (no page dedup)
    store = PineconeStore()
    raw_matches = store.query_raw(
        query_embedding, fetch_k=RERANK_FETCH_K, catalog_filter=catalog_filter
    )

    # Re-rank using embedding score aggregation, tile coverage, and color similarity
    results = rerank(raw_matches, query_image, top_k=top_k)

    return {"results": results, "total": len(results)}

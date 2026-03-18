from __future__ import annotations

"""
fab-search — Search API endpoint.
"""

import io
from fastapi import APIRouter, UploadFile, File, Query
from PIL import Image

from src.config import TOP_K
from src.embedding_client import encode_image
from src.pinecone_store import PineconeStore

router = APIRouter()


@router.post("/api/search")
async def search(
    file: UploadFile = File(..., description="Query image (jpg/png/webp)"),
    top_k: int = Query(default=TOP_K, ge=1, le=20, description="Number of results"),
):
    """
    Search the catalog index with an uploaded image.

    Returns ranked results with page image URLs and highlight coordinates.
    """
    contents = await file.read()
    query_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Get embedding from HF Space
    query_embedding = encode_image(query_image)

    # Query Pinecone
    store = PineconeStore()
    results = store.query(query_embedding, top_k=top_k)

    return {"results": results, "total": len(results)}

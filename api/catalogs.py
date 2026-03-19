from __future__ import annotations

"""
fab-search — Catalog management API endpoints.
"""

from fastapi import APIRouter, HTTPException

from src.pinecone_store import PineconeStore
from src.blob_store import delete_catalog_blobs

router = APIRouter()


@router.get("/api/catalogs")
async def list_catalogs():
    """List all indexed catalogs with page/vector counts."""
    store = PineconeStore()
    catalogs = store.list_catalogs()
    return {"catalogs": catalogs, "total": len(catalogs)}


@router.delete("/api/catalogs/{catalog_name:path}")
async def remove_catalog(catalog_name: str):
    """Remove a catalog from the Pinecone index."""
    store = PineconeStore()

    if not store.has_catalog(catalog_name):
        raise HTTPException(status_code=404, detail=f"Catalog '{catalog_name}' not found")

    blob_urls = store.get_catalog_blob_urls(catalog_name)
    store.remove_catalog(catalog_name)
    deleted = delete_catalog_blobs(blob_urls)
    return {"message": f"Catalog '{catalog_name}' removed ({deleted} blobs deleted)", "catalog_name": catalog_name}

from __future__ import annotations

"""
fab-search — Catalog management API endpoints.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.pinecone_store import PineconeStore
from src.blob_store import delete_catalog_blobs

log = logging.getLogger(__name__)
router = APIRouter()


class CatalogAction(BaseModel):
    action: str
    catalog_name: str


@router.get("/api/catalogs")
async def list_catalogs():
    """List all indexed catalogs with page/vector counts."""
    store = PineconeStore()
    catalogs = store.list_catalogs()
    return {"catalogs": catalogs, "total": len(catalogs)}


@router.post("/api/catalogs")
async def manage_catalog(body: CatalogAction):
    """Manage catalogs. Supports action='delete'."""
    if body.action != "delete":
        raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")

    store = PineconeStore()

    log.info("manage_catalog: action=%r, catalog_name=%r", body.action, body.catalog_name)

    has_vectors = store.has_catalog(body.catalog_name)
    blob_urls = store.get_catalog_blob_urls(body.catalog_name) if has_vectors else []

    if not has_vectors and not blob_urls:
        return {
            "message": f"Catalog '{body.catalog_name}' already deleted",
            "catalog_name": body.catalog_name,
        }

    deleted_blobs = 0
    if blob_urls:
        deleted_blobs = delete_catalog_blobs(blob_urls)
    if has_vectors:
        store.remove_catalog(body.catalog_name)

    return {
        "message": f"Catalog '{body.catalog_name}' removed ({deleted_blobs} blobs deleted)",
        "catalog_name": body.catalog_name,
    }

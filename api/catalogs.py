from __future__ import annotations

"""
fab-search — Catalog management API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.pinecone_store import PineconeStore
from src.blob_store import delete_catalog_blobs

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

    if not store.has_catalog(body.catalog_name):
        raise HTTPException(status_code=404, detail=f"Catalog '{body.catalog_name}' not found")

    blob_urls = store.get_catalog_blob_urls(body.catalog_name)
    store.remove_catalog(body.catalog_name)
    deleted = delete_catalog_blobs(blob_urls)
    return {"message": f"Catalog '{body.catalog_name}' removed ({deleted} blobs deleted)", "catalog_name": body.catalog_name}

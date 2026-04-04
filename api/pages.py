from __future__ import annotations

"""
fab-search — Page browsing API endpoints.

Provides endpoints to browse catalog pages and inspect tile regions
used during indexing.
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from src.pinecone_store import PineconeStore

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/pages")
async def get_pages(
    catalog: str = Query(..., description="PDF catalog name"),
    page: int | None = Query(None, description="1-based page number", ge=1),
):
    """
    Browse catalog pages and their tile regions.

    Without `page`: returns sorted list of page numbers for the catalog.
    With `page`: returns the page image URL and all tile bounding boxes.
    """
    store = PineconeStore()

    if page is None:
        pages = store.get_catalog_pages(catalog)
        if not pages:
            raise HTTPException(status_code=404, detail=f"Catalog '{catalog}' not found")
        return {
            "pdf_name": catalog,
            "pages": pages,
            "total_pages": len(pages),
        }

    data = store.get_page_tiles(catalog, page)
    if not data["tiles"]:
        raise HTTPException(
            status_code=404,
            detail=f"No tiles found for '{catalog}' page {page}",
        )
    return {
        "pdf_name": catalog,
        "page_number": page,
        "page_image": data["page_image"],
        "tiles": data["tiles"],
        "total_tiles": len(data["tiles"]),
    }

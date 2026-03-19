from __future__ import annotations

"""
fab-search — Vercel Blob storage client for page images.

Uploads rendered page PNGs to Vercel Blob so they can be served
via public URLs in search results.

Uses the Vercel Blob REST API with a read-write token.
"""

import io
import logging
import httpx
from PIL import Image

from src.config import BLOB_READ_WRITE_TOKEN

_BLOB_API_URL = "https://blob.vercel-storage.com"

log = logging.getLogger(__name__)


def upload_page_image(
    pil_image: Image.Image,
    filename: str,
) -> str:
    """
    Upload a page image to Vercel Blob.

    Args:
        pil_image:  PIL Image to upload.
        filename:   Storage path/name (e.g. "pages/catalog_page_0001.png").

    Returns:
        Public URL of the uploaded blob.
    """
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    blob_bytes = buf.getvalue()

    response = httpx.put(
        f"{_BLOB_API_URL}/{filename}",
        content=blob_bytes,
        headers={
            "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
            "x-content-type": "image/jpeg",
            "x-cache-control-max-age": "31536000",
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Blob upload failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    return data["url"]


def delete_catalog_blobs(blob_urls: list[str]) -> int:
    """
    Delete blobs from Vercel Blob storage by their URLs.

    Args:
        blob_urls:  List of public blob URLs to delete.

    Returns:
        Number of blobs successfully requested for deletion.
    """
    if not blob_urls:
        return 0

    # Vercel Blob delete API accepts up to 1000 URLs per request
    deleted = 0
    batch_size = 1000
    for i in range(0, len(blob_urls), batch_size):
        batch = blob_urls[i : i + batch_size]
        response = httpx.post(
            f"{_BLOB_API_URL}/delete",
            json={"urls": batch},
            headers={
                "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
            },
            timeout=60.0,
        )
        if response.status_code == 200:
            deleted += len(batch)
        else:
            log.warning(
                "Blob delete failed (%d): %s", response.status_code, response.text
            )

    return deleted

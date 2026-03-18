from __future__ import annotations

"""
fab-search — Vercel Blob storage client for page images.

Uploads rendered page PNGs to Vercel Blob so they can be served
via public URLs in search results.

Uses the Vercel Blob REST API with a read-write token.
"""

import io
import httpx
from PIL import Image

from src.config import BLOB_READ_WRITE_TOKEN

_BLOB_API_URL = "https://blob.vercel-storage.com"


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
    pil_image.save(buf, format="PNG")
    blob_bytes = buf.getvalue()

    response = httpx.put(
        f"{_BLOB_API_URL}/{filename}",
        content=blob_bytes,
        headers={
            "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
            "x-content-type": "image/png",
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

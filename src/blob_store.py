from __future__ import annotations

"""
fab-search — Cloudflare R2 storage client for page images.

Uploads rendered page JPEGs to Cloudflare R2 so they can be served
via public URLs in search results.

Uses the S3-compatible API via boto3.
"""

import io
import logging
import time
import boto3
from botocore.exceptions import ClientError
from PIL import Image

from src.config import (
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME, R2_PUBLIC_URL,
)

log = logging.getLogger(__name__)


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def upload_page_image(
    pil_image: Image.Image,
    filename: str,
) -> str:
    """
    Upload a page image to Cloudflare R2.

    Args:
        pil_image:  PIL Image to upload.
        filename:   Storage path/name (e.g. "pages/catalog_page_0001.png").

    Returns:
        Public URL of the uploaded object.
    """
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    buf.seek(0)

    s3 = _get_s3_client()

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            s3.put_object(
                Bucket=R2_BUCKET_NAME,
                Key=filename,
                Body=buf.getvalue(),
                ContentType="image/jpeg",
                CacheControl="public, max-age=31536000",
            )
            break
        except ClientError as e:
            if attempt < max_retries:
                log.warning(
                    "R2 upload failed (attempt %d/%d): %s — retrying in 10s...",
                    attempt, max_retries, e,
                )
                time.sleep(10)
                continue
            raise RuntimeError(f"R2 upload failed after {max_retries} attempts: {e}") from e

    return f"{R2_PUBLIC_URL}/{filename}"


def delete_catalog_blobs(blob_urls: list[str]) -> int:
    """
    Delete objects from Cloudflare R2 by their public URLs.

    Args:
        blob_urls:  List of public URLs to delete.

    Returns:
        Number of objects successfully requested for deletion.
    """
    if not blob_urls:
        return 0

    s3 = _get_s3_client()
    prefix = R2_PUBLIC_URL.rstrip("/") + "/"

    # Convert public URLs to R2 object keys
    keys = []
    for url in blob_urls:
        if url.startswith(prefix):
            keys.append(url[len(prefix):])
        else:
            log.warning("Skipping unrecognised blob URL: %s", url)

    if not keys:
        return 0

    # S3 delete_objects accepts up to 1000 keys per request
    deleted = 0
    batch_size = 1000
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        try:
            s3.delete_objects(
                Bucket=R2_BUCKET_NAME,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
            deleted += len(batch)
        except ClientError as e:
            log.warning("R2 batch delete failed: %s", e)

    return deleted

from __future__ import annotations

"""
fab-search — Embedding client using direct HTTP requests to
the Gradio API on HuggingFace Spaces.

Uses the two-step Gradio curl API:
  1. POST /gradio_api/call/get_embedding  → EVENT_ID
  2. GET  /gradio_api/call/get_embedding/{EVENT_ID}  → SSE stream with result
"""

import io
import json
import logging
import tempfile
import time
import os

import numpy as np
import requests
from PIL import Image

from src.config import HF_SPACE_ID

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 120  # seconds

# Derive the base URL from the Space ID  (owner/name → owner-name.hf.space)
_BASE_URL: str | None = None


def _get_base_url() -> str:
    global _BASE_URL
    if _BASE_URL is None:
        space_id = HF_SPACE_ID  # e.g. "knayirp/fab-search"
        slug = space_id.replace("/", "-")
        _BASE_URL = f"https://{slug}.hf.space"
        log.info("HF Space base URL: %s", _BASE_URL)
    return _BASE_URL


def _upload_file(img_bytes: bytes, filename: str = "image.png") -> str:
    """Upload a file to the Space's Gradio upload endpoint, return the server path."""
    base = _get_base_url()
    url = f"{base}/gradio_api/upload"
    log.debug("Uploading %d bytes to %s", len(img_bytes), url)

    resp = requests.post(
        url,
        files={"files": (filename, img_bytes, "image/png")},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    paths = resp.json()  # list of server-side file paths
    log.debug("Upload response: %s", paths)
    return paths[0]


def _call_endpoint(server_path: str) -> str:
    """
    Call /get_embedding via the two-step Gradio API.
    Returns the raw result string (JSON).
    """
    base = _get_base_url()
    submit_url = f"{base}/gradio_api/call/get_embedding"

    # Step 1: POST to submit the job
    payload = {
        "data": [
            {"path": server_path, "meta": {"_type": "gradio.FileData"}}
        ]
    }
    log.debug("POST %s payload=%s", submit_url, json.dumps(payload)[:300])
    resp = requests.post(
        submit_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    event_id = resp.json().get("event_id")
    if not event_id:
        raise RuntimeError(f"No event_id in submit response: {resp.text[:300]}")
    log.debug("Got event_id: %s", event_id)

    # Step 2: GET the SSE stream to fetch the result
    result_url = f"{submit_url}/{event_id}"
    log.debug("GET %s", result_url)
    resp = requests.get(result_url, stream=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # Parse Server-Sent Events — look for "event: complete" followed by "data: ..."
    event_type = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_type == "complete":
            data_str = line.split(":", 1)[1].strip()
            log.debug("SSE complete data (first 300 chars): %.300s", data_str)
            # data is a JSON array; the first element is our embedding JSON string
            outer = json.loads(data_str)
            return outer[0]
        elif line.startswith("data:") and event_type == "error":
            raise RuntimeError(f"Space returned error: {line}")

    raise RuntimeError("SSE stream ended without a 'complete' event")


def encode_image(pil_image: Image.Image) -> np.ndarray:
    """
    Encode a PIL image into a normalized CLIP embedding via the HF Space.

    Returns:
        1-D float32 numpy array (512-dim for ViT-B-32).
    """
    img = pil_image.convert("RGB")
    log.debug("Image to encode: size=%s", img.size)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    log.debug("PNG size: %d bytes", len(img_bytes))

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            server_path = _upload_file(img_bytes)
            log.info("Calling /get_embedding (attempt %d, image %dx%d)...", attempt, img.width, img.height)
            result = _call_endpoint(server_path)
            break
        except (requests.RequestException, RuntimeError, json.JSONDecodeError) as e:
            last_err = e
            log.warning(
                "Attempt %d/%d failed: %s. Retrying in %ds...",
                attempt, MAX_RETRIES, e, RETRY_DELAY,
                exc_info=True,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    else:
        raise RuntimeError(
            f"Embedding failed after {MAX_RETRIES} attempts. Last error: {last_err}"
        )

    # Parse the embedding JSON string
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        log.error("Failed to parse embedding JSON. Raw: %.500s", result)
        raise RuntimeError(f"Embedding response is not valid JSON: {str(result)[:200]}")

    if "embedding" not in data:
        log.error("No 'embedding' key. Keys: %s. Raw: %.500s", list(data.keys()), result)
        raise RuntimeError(f"Missing 'embedding' key, got: {list(data.keys())}")

    embedding = np.array(data["embedding"], dtype=np.float32)
    log.debug("Embedding shape=%s", embedding.shape)

    norm = np.linalg.norm(embedding)
    if norm > 0 and abs(norm - 1.0) > 1e-3:
        log.debug("Re-normalizing (norm was %.4f)", norm)
        embedding = embedding / norm

    return embedding


def encode_image_batch(images: list[Image.Image]) -> list[np.ndarray]:
    """Encode multiple images sequentially."""
    return [encode_image(img) for img in images]

from __future__ import annotations

"""
fab-search — Embedding client using the Gradio API on HuggingFace Spaces.

Calls the knayirp/fab-search Space's /get_embedding endpoint via
gradio_client.  The Space serves ViT-B-32 (laion2b_s34b_b79k) and
returns the 512-dim L2-normalised CLIP embedding as a JSON string.
"""

import json
import tempfile
import os
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file

from src.config import HF_SPACE_ID

# Reuse a single Gradio client across calls
_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        hf_token = os.environ.get("HF_TOKEN") or None
        _client = Client(HF_SPACE_ID, hf_token=hf_token)
    return _client


def encode_image(pil_image: Image.Image) -> np.ndarray:
    """
    Encode a PIL image into a normalized CLIP embedding via the HF Space.

    Args:
        pil_image:  PIL Image (any mode — will be converted to RGB).

    Returns:
        1-D float32 numpy array (512-dim for ViT-B-32).

    Raises:
        RuntimeError: If the embedding service returns an error.
    """
    img = pil_image.convert("RGB")

    # gradio_client needs a file path; write to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        img.save(tmp, format="PNG")
        tmp.close()

        client = _get_client()
        result = client.predict(
            image=handle_file(tmp.name),
            api_name="/get_embedding",
        )
    finally:
        os.unlink(tmp.name)

    # result is a JSON string from the Gradio Textbox
    data = json.loads(result)
    embedding = np.array(data["embedding"], dtype=np.float32)

    # Re-normalize if needed
    norm = np.linalg.norm(embedding)
    if norm > 0 and abs(norm - 1.0) > 1e-3:
        embedding = embedding / norm

    return embedding


def encode_image_batch(images: list[Image.Image]) -> list[np.ndarray]:
    """
    Encode multiple images sequentially.
    """
    return [encode_image(img) for img in images]

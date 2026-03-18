from __future__ import annotations

"""
fab-search — PDF to page image conversion using PyMuPDF.

Replaces pdf2image + Poppler from FabricSearch with pymupdf (fitz),
which bundles MuPDF binaries in the pip wheel — no system dependencies.
"""

import os
import fitz  # pymupdf
from PIL import Image


def render_pdf(pdf_path: str, dpi: int, pages_dir: str) -> list[tuple[int, str, Image.Image]]:
    """
    Render every page of a PDF to a PIL Image and save to disk.

    Args:
        pdf_path:   Path to the PDF file.
        dpi:        Resolution to render at.
        pages_dir:  Directory to save page images.

    Returns:
        List of (page_number, page_image_path, pil_image).
        page_number is 1-based.
    """
    os.makedirs(pages_dir, exist_ok=True)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    results = []
    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1  # 1-based

        # Render at the requested DPI (default PDF is 72 DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        filename = f"{pdf_name}_page_{page_num:04d}.png"
        filepath = os.path.join(pages_dir, filename)
        img.save(filepath, "PNG")

        results.append((page_num, filepath, img))

    doc.close()
    return results


def render_pdf_pages(pdf_path: str, dpi: int) -> list[tuple[int, Image.Image]]:
    """
    Render PDF pages to PIL Images without saving to disk.

    Useful when page images will be uploaded to cloud storage separately.

    Returns:
        List of (page_number, pil_image). page_number is 1-based.
    """
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    results = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        results.append((i + 1, img))

    doc.close()
    return results

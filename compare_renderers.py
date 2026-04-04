"""
Diagnostic: compare MuPDF vs Poppler PDF rendering and their effect
on contour-based segmentation.

Prerequisites:
    pip install pdf2image
    Poppler must be on PATH (Windows: download from
    https://github.com/osber/poppler/releases and add bin/ to PATH)

Usage:
    python compare_renderers.py <pdf_path>
    python compare_renderers.py <pdf_path> --page 3 --debug
    python compare_renderers.py <pdf_path> --page 1-5 --debug
"""

import argparse
import os
import sys
import time

import fitz  # pymupdf
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

from src.config import (
    DPI,
    MAX_REGION_AREA_FRAC,
    MERGE_IOU_THRESHOLD,
    MIN_REGION_AREA_FRAC,
    MORPH_KERNEL_SIZE,
    REGION_PADDING,
)
from src.segment import _detect_regions_impl


# ── Renderers ─────────────────────────────────────────────────

def render_mupdf(pdf_path: str, page_num: int, dpi: int) -> Image.Image:
    """Render a single page using MuPDF (fitz)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def render_poppler(pdf_path: str, page_num: int, dpi: int) -> Image.Image:
    """Render a single page using Poppler via pdf2image."""
    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=page_num, last_page=page_num,
    )
    return images[0]


# ── Visualization helpers ─────────────────────────────────────

def draw_boxes(img: Image.Image, boxes: list, label: str) -> Image.Image:
    """Draw bounding boxes and a label on a copy of the image."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)

    # Label in top-left
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, 500, 50], fill="black")
    draw.text((10, 8), label, fill="lime", font=font)
    return annotated


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Stitch two images horizontally with a 4px red divider."""
    gap = 4
    h = max(left.height, right.height)
    combined = Image.new("RGB", (left.width + gap + right.width, h), "red")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + gap, 0))
    return combined


def pixel_diff_stats(a: Image.Image, b: Image.Image) -> dict:
    """Compute pixel-level difference statistics between two images."""
    # Resize to common dimensions if they differ
    if a.size != b.size:
        common_w = min(a.width, b.width)
        common_h = min(a.height, b.height)
        a = a.crop((0, 0, common_w, common_h))
        b = b.crop((0, 0, common_w, common_h))
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    diff = np.abs(arr_a - arr_b)
    return {
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "nonzero_pct": float((diff > 0).any(axis=-1).sum() / diff[:, :, 0].size * 100),
    }


# ── Main comparison ──────────────────────────────────────────

def compare_page(pdf_path: str, page_num: int, dpi: int, out_dir: str, debug: bool):
    """Run the full comparison for one page."""
    seg_kwargs = dict(
        min_area_frac=MIN_REGION_AREA_FRAC,
        max_area_frac=MAX_REGION_AREA_FRAC,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        merge_iou_threshold=MERGE_IOU_THRESHOLD,
        padding=REGION_PADDING,
    )

    # 1. Render with both engines
    print(f"\n{'=' * 60}")
    print(f"Page {page_num}")
    print(f"{'=' * 60}")

    t = time.time()
    img_mupdf = render_mupdf(pdf_path, page_num, dpi)
    t_mupdf = time.time() - t

    t = time.time()
    img_poppler = render_poppler(pdf_path, page_num, dpi)
    t_poppler = time.time() - t

    print(f"  MuPDF   : {img_mupdf.size[0]}x{img_mupdf.size[1]}  ({t_mupdf:.2f}s)")
    print(f"  Poppler : {img_poppler.size[0]}x{img_poppler.size[1]}  ({t_poppler:.2f}s)")

    # 2. Pixel diff
    stats = pixel_diff_stats(img_mupdf, img_poppler)
    print(f"  Pixel diff: mean={stats['mean_diff']:.2f}  max={stats['max_diff']:.0f}  "
          f"changed={stats['nonzero_pct']:.1f}%")

    # 3. Run segmentation on both
    boxes_m, binary_m, morphed_m = _detect_regions_impl(img_mupdf, **seg_kwargs)
    boxes_p, binary_p, morphed_p = _detect_regions_impl(img_poppler, **seg_kwargs)

    print(f"  Regions detected  MuPDF: {len(boxes_m)}   Poppler: {len(boxes_p)}")

    if len(boxes_m) != len(boxes_p):
        print(f"  *** MISMATCH — region counts differ ***")

    # Print per-region boxes
    for tag, boxes in [("MuPDF", boxes_m), ("Poppler", boxes_p)]:
        for i, (x, y, w, h) in enumerate(boxes):
            print(f"    {tag} region {i+1}: x={x} y={y} w={w} h={h}")

    # 4. Annotated images
    ann_m = draw_boxes(img_mupdf, boxes_m, f"MuPDF ({len(boxes_m)} regions)")
    ann_p = draw_boxes(img_poppler, boxes_p, f"Poppler ({len(boxes_p)} regions)")

    combined = side_by_side(ann_m, ann_p)
    out_path = os.path.join(out_dir, f"page_{page_num:04d}_comparison.png")
    combined.save(out_path)
    print(f"  Saved: {out_path}")

    # 5. Debug images
    if debug:
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        Image.fromarray(binary_m).save(
            os.path.join(debug_dir, f"page_{page_num:04d}_mupdf_binary.png"))
        Image.fromarray(morphed_m).save(
            os.path.join(debug_dir, f"page_{page_num:04d}_mupdf_morphed.png"))
        Image.fromarray(binary_p).save(
            os.path.join(debug_dir, f"page_{page_num:04d}_poppler_binary.png"))
        Image.fromarray(morphed_p).save(
            os.path.join(debug_dir, f"page_{page_num:04d}_poppler_morphed.png"))

        # Raw renders (un-annotated)
        img_mupdf.save(
            os.path.join(debug_dir, f"page_{page_num:04d}_mupdf_raw.png"))
        img_poppler.save(
            os.path.join(debug_dir, f"page_{page_num:04d}_poppler_raw.png"))

        print(f"  Debug images saved to {debug_dir}/")


def parse_page_range(s: str, max_pages: int) -> list[int]:
    """Parse '3', '1-5', or 'all' into a list of 1-based page numbers."""
    if s.lower() == "all":
        return list(range(1, max_pages + 1))
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), min(int(hi), max_pages) + 1))
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(
        description="Compare MuPDF vs Poppler rendering + segmentation",
    )
    parser.add_argument("pdf_path", help="Path to a PDF catalog")
    parser.add_argument("--page", default="1",
                        help="Page number, range (1-5), or 'all' (default: 1)")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help=f"Render DPI (default: {DPI})")
    parser.add_argument("--out", default="./compare_output",
                        help="Output directory (default: ./compare_output)")
    parser.add_argument("--debug", action="store_true",
                        help="Save binary/morphed/raw debug images")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: {args.pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # Get total page count from MuPDF (fast)
    doc = fitz.open(args.pdf_path)
    total_pages = len(doc)
    doc.close()

    pages = parse_page_range(args.page, total_pages)
    print(f"PDF: {args.pdf_path}  ({total_pages} pages)")
    print(f"Comparing pages: {pages}  DPI: {args.dpi}")

    for p in pages:
        compare_page(args.pdf_path, p, args.dpi, args.out, args.debug)

    print(f"\nDone. Results in {args.out}/")


if __name__ == "__main__":
    main()

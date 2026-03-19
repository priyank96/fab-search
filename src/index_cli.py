from __future__ import annotations

"""
fab-search — Indexing CLI.

Standalone script for indexing PDF catalogs into Pinecone via the
HuggingFace Space embedding endpoint. Runs locally or on a worker —
NOT on Vercel (too slow for serverless timeout).

Usage:
    python -m src.index_cli list
    python -m src.index_cli add  <path_to_pdf>
    python -m src.index_cli remove <catalog_name>
    python -m src.index_cli rebuild
"""

import logging
import os
import sys
import glob
import time
import io
import argparse

from src.config import (
    CATALOGS_DIR, PAGES_DIR, DPI,
    MIN_REGION_AREA_FRAC, MAX_REGION_AREA_FRAC,
    MORPH_KERNEL_SIZE, MERGE_IOU_THRESHOLD, REGION_PADDING,
    TILE_FALLBACK_MIN_REGIONS,
    TILE_SIZE, STRIDE, BLANK_THRESHOLD, WHITE_VALUE,
)
from src.embedding_client import encode_image
from src.pdf_processor import render_pdf
from src.segment import detect_regions
from src.tiler import generate_filtered_tiles
from src.pinecone_store import PineconeStore
from src.blob_store import upload_page_image, delete_catalog_blobs


def index_single_pdf(pdf_path: str, store: PineconeStore) -> tuple[int, int, int]:
    """
    Run the indexing pipeline for a single PDF.

    Returns:
        (num_pages, num_regions, num_vectors)
    """
    pdf_name = os.path.basename(pdf_path)

    # Render PDF pages to disk (for later reference / upload)
    t1 = time.time()
    pages = render_pdf(pdf_path, DPI, PAGES_DIR)
    print(f"    Rendered {len(pages)} pages in {time.time() - t1:.1f}s")

    pdf_pages = len(pages)
    pdf_regions = 0
    pdf_vectors = 0

    for page_num, page_image_path, page_img in pages:
        # Upload page image to Vercel Blob for public access
        blob_filename = f"pages/{pdf_name}/{os.path.basename(page_image_path)}"
        page_image_url = upload_page_image(page_img, blob_filename)
        page_w, page_h = page_img.size

        # Full-page embedding (always added)
        t_enc = time.time()
        page_embedding = encode_image(page_img)
        store.add(
            embedding=page_embedding,
            pdf_name=pdf_name,
            page_number=page_num,
            tile_x=0,
            tile_y=0,
            tile_w=page_w,
            tile_h=page_h,
            page_image=page_image_url,
        )
        page_vectors = 1

        # Detect pattern regions via contour segmentation
        regions = detect_regions(
            page_img,
            min_area_frac=MIN_REGION_AREA_FRAC,
            max_area_frac=MAX_REGION_AREA_FRAC,
            morph_kernel_size=MORPH_KERNEL_SIZE,
            merge_iou_threshold=MERGE_IOU_THRESHOLD,
            padding=REGION_PADDING,
        )

        if len(regions) < TILE_FALLBACK_MIN_REGIONS:
            # Fallback: sliding window tiling
            tiles, raw_count, blank_count = generate_filtered_tiles(
                page_img, TILE_SIZE, STRIDE, BLANK_THRESHOLD, WHITE_VALUE,
            )
            pdf_regions += len(tiles)

            for tx, ty, tile_img in tiles:
                tw, th = tile_img.size
                tile_embedding = encode_image(tile_img)
                store.add(
                    embedding=tile_embedding,
                    pdf_name=pdf_name,
                    page_number=page_num,
                    tile_x=tx,
                    tile_y=ty,
                    tile_w=tw,
                    tile_h=th,
                    page_image=page_image_url,
                )
                page_vectors += 1

            print(
                f"    Page {page_num:>3d}: "
                f"0 regions -> TILE FALLBACK "
                f"({len(tiles)} tiles, {blank_count} blank skipped) + 1 page -> "
                f"{page_vectors:>3d} vectors"
            )
        else:
            # Normal: encode each detected region
            pdf_regions += len(regions)

            for rx, ry, rw, rh in regions:
                region_img = page_img.crop((rx, ry, rx + rw, ry + rh))
                region_embedding = encode_image(region_img)
                store.add(
                    embedding=region_embedding,
                    pdf_name=pdf_name,
                    page_number=page_num,
                    tile_x=rx,
                    tile_y=ry,
                    tile_w=rw,
                    tile_h=rh,
                    page_image=page_image_url,
                )
                page_vectors += 1

            print(
                f"    Page {page_num:>3d}: "
                f"{len(regions):>3d} regions detected + 1 page -> "
                f"{page_vectors:>3d} vectors"
            )

        pdf_vectors += page_vectors

        # Flush after each page so progress is saved incrementally
        store.flush()

    # Flush remaining vectors to Pinecone
    store.flush()
    return pdf_pages, pdf_regions, pdf_vectors


# ── Subcommands ───────────────────────────────────────────────

def cmd_list(_args):
    """List indexed catalogs."""
    store = PineconeStore()
    catalogs = store.list_catalogs()

    if not catalogs:
        print("Index is empty — no catalogs indexed.")
        return

    print(f"\n{'Catalog':<45s} {'Pages':>6s} {'Vectors':>8s}")
    print("-" * 61)
    total_pages = 0
    total_vectors = 0
    for c in catalogs:
        print(f"{c['pdf_name']:<45s} {c['num_pages']:>6d} {c['num_vectors']:>8d}")
        total_pages += c["num_pages"]
        total_vectors += c["num_vectors"]
    print("-" * 61)
    print(f"{'TOTAL':<45s} {total_pages:>6d} {total_vectors:>8d}")
    print(f"\n{len(catalogs)} catalog(s) in index.")


def cmd_add(args):
    """Add (or re-index) a single catalog."""
    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    pdf_name = os.path.basename(pdf_path)
    store = PineconeStore()

    # Remove old version if it exists
    if store.has_catalog(pdf_name):
        blob_urls = store.get_catalog_blob_urls(pdf_name)
        deleted_blobs = delete_catalog_blobs(blob_urls)
        store.remove_catalog(pdf_name)
        print(f"Removed previous version of '{pdf_name}' ({deleted_blobs} blobs deleted)")

    os.makedirs(PAGES_DIR, exist_ok=True)

    print(f"\nIndexing: {pdf_name}")
    t0 = time.time()
    pages, regions, vectors = index_single_pdf(pdf_path, store)
    elapsed = time.time() - t0

    store.close()

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Pages:   {pages}")
    print(f"  Regions: {regions}")
    print(f"  Vectors: {vectors}")


def cmd_remove(args):
    """Remove a catalog from the index."""
    catalog_name = args.catalog_name
    store = PineconeStore()

    has_vectors = store.has_catalog(catalog_name)
    blob_urls = store.get_catalog_blob_urls(catalog_name) if has_vectors else []

    if not has_vectors:
        catalogs = store.list_catalogs()
        print(f"Catalog '{catalog_name}' not found in index.")
        if catalogs:
            print("Indexed catalogs:")
            for c in catalogs:
                print(f"  - {c['pdf_name']}")
        return

    deleted_blobs = delete_catalog_blobs(blob_urls)
    store.remove_catalog(catalog_name)
    print(f"Removed '{catalog_name}' from index ({deleted_blobs} blobs deleted).")


def cmd_rebuild(_args):
    """Full rebuild of the index from all PDFs in catalogs/."""
    pdf_files = sorted(glob.glob(os.path.join(CATALOGS_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {CATALOGS_DIR}")
        return

    store = PineconeStore()

    # Remove all existing catalogs first
    existing = store.list_catalogs()
    for cat in existing:
        blob_urls = store.get_catalog_blob_urls(cat["pdf_name"])
        deleted_blobs = delete_catalog_blobs(blob_urls)
        store.remove_catalog(cat["pdf_name"])
        print(f"Removed existing catalog: {cat['pdf_name']} ({deleted_blobs} blobs deleted)")

    os.makedirs(PAGES_DIR, exist_ok=True)

    print(f"\nRebuilding index from {len(pdf_files)} catalog(s)...")
    total_pages = 0
    total_regions = 0
    total_vectors = 0
    pipeline_start = time.time()

    for pdf_idx, pdf_path in enumerate(pdf_files):
        pdf_name = os.path.basename(pdf_path)
        print(f"\n  [{pdf_idx + 1}/{len(pdf_files)}] {pdf_name}")

        pages, regions, vectors = index_single_pdf(pdf_path, store)
        total_pages += pages
        total_regions += regions
        total_vectors += vectors

    store.close()

    elapsed = time.time() - pipeline_start
    print(f"\nIndexing complete in {elapsed:.1f}s")
    print(f"  PDFs:    {len(pdf_files)}")
    print(f"  Pages:   {total_pages}")
    print(f"  Regions: {total_regions}")
    print(f"  Vectors: {total_vectors}")


# ── CLI entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="index_cli",
        description="fab-search — Index PDF catalogs into Pinecone",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="Show indexed catalogs")

    p_add = sub.add_parser("add", help="Add or re-index a single catalog PDF")
    p_add.add_argument("pdf", help="Path to the PDF file to index")

    p_rm = sub.add_parser("remove", help="Remove a catalog from the index")
    p_rm.add_argument(
        "catalog_name",
        help='PDF filename to remove (e.g. "catalog.pdf")',
    )

    sub.add_parser("rebuild", help="Full rebuild from all catalogs/ PDFs")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "add": cmd_add,
        "remove": cmd_remove,
        "rebuild": cmd_rebuild,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Only show debug logs from the embedding client
    logging.getLogger("src.embedding_client").setLevel(logging.DEBUG)
    main()

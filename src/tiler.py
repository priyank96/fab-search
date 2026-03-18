from __future__ import annotations

"""
fab-search — Sliding window tiling and blank tile filtering.

Ported from FabricSearch with no changes (pure PIL + numpy, no system deps).
"""

import numpy as np
from PIL import Image


def generate_tiles(
    page_image: Image.Image,
    tile_size: int,
    stride: int,
) -> list[tuple[int, int, Image.Image]]:
    """
    Generate tiles from a page image using a sliding window.

    Returns:
        List of (x, y, tile_image) where x, y are the top-left pixel
        coordinates of the tile on the page.
    """
    width, height = page_image.size
    tiles = []

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = page_image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append((x, y, tile))

    return tiles


def is_blank(
    tile_image: Image.Image,
    threshold: float = 0.90,
    white_value: int = 240,
) -> bool:
    """
    Check if a tile is mostly blank (near-white).
    """
    arr = np.array(tile_image)
    near_white = np.all(arr > white_value, axis=-1)
    fraction = near_white.sum() / near_white.size
    return fraction >= threshold


def generate_filtered_tiles(
    page_image: Image.Image,
    tile_size: int,
    stride: int,
    blank_threshold: float = 0.90,
    white_value: int = 240,
) -> tuple[list[tuple[int, int, Image.Image]], int, int]:
    """
    Generate tiles and filter out blank ones.

    Returns:
        (filtered_tiles, raw_count, blank_count)
    """
    tiles = generate_tiles(page_image, tile_size, stride)
    raw_count = len(tiles)
    filtered = [
        (x, y, tile)
        for x, y, tile in tiles
        if not is_blank(tile, blank_threshold, white_value)
    ]
    blank_count = raw_count - len(filtered)
    return filtered, raw_count, blank_count

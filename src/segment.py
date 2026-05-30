from __future__ import annotations

"""
fab-search — Contour-based pattern segmentation.

Detects individual fabric pattern regions on a catalog page using
OpenCV contour detection. Ported from FabricSearch with no algorithm changes.

Requires opencv-python-headless (no GUI dependencies).
"""

import cv2
import numpy as np
from PIL import Image


def _detect_regions_impl(
    page_image: Image.Image,
    min_area_frac: float = 0.005,
    max_area_frac: float = 0.80,
    morph_kernel_size: int = 15,
    merge_iou_threshold: float = 0.30,
    padding: int = 10,
    trim_label_bands: bool = True,
    label_band_max_frac: float = 0.35,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    """
    Internal implementation of contour-based region detection.

    Returns:
        (boxes, binary_image, morphed_image)
    """
    img_rgb = np.array(page_image)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    page_h, page_w = gray.shape
    page_area = page_h * page_w
    min_area = int(page_area * min_area_frac)
    max_area = int(page_area * max_area_frac)

    # 1. Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    block_size = max(3, (min(page_w, page_h) // 20) | 1)  # ensure odd
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, 10,
    )

    # 2. Morphological closing
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.dilate(closed, dilate_kernel, iterations=1)

    # 3. Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4. Filter by area
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area or area > max_area:
            continue

        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 8.0:
            continue

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(page_w - x, w + 2 * padding)
        h = min(page_h - y, h + 2 * padding)

        # Trim solid label bars / blank page margins at the top & bottom edges.
        if trim_label_bands:
            top_trim, bot_trim = _trim_label_bands(
                gray[y : y + h, x : x + w], label_band_max_frac
            )
            if top_trim or bot_trim:
                y += top_trim
                h -= top_trim + bot_trim

        boxes.append((x, y, w, h))

    # 5. Merge overlapping boxes
    if boxes:
        boxes = _merge_overlapping(boxes, merge_iou_threshold)
        boxes.sort(key=lambda b: (b[1], b[0]))

    return boxes, binary, closed


def detect_regions(
    page_image: Image.Image,
    min_area_frac: float = 0.005,
    max_area_frac: float = 0.80,
    morph_kernel_size: int = 15,
    merge_iou_threshold: float = 0.30,
    padding: int = 10,
    trim_label_bands: bool = True,
    label_band_max_frac: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    """
    Detect fabric pattern regions on a catalog page.

    Returns:
        List of (x, y, w, h) bounding boxes sorted top-to-bottom, left-to-right.
    """
    boxes, _, _ = _detect_regions_impl(
        page_image, min_area_frac, max_area_frac,
        morph_kernel_size, merge_iou_threshold, padding,
        trim_label_bands, label_band_max_frac,
    )
    return boxes


def _trim_label_bands(
    gray_crop: np.ndarray,
    max_band_frac: float = 0.35,
    ratio: float = 0.5,
    min_body_std: float = 6.0,
    min_band_px: int = 4,
) -> tuple[int, int]:
    """
    Detect contiguous low-texture horizontal bands at the top/bottom edges of a
    region crop and return the number of pixels to trim from each.

    Catalog label bars (solid colour with a line of text) and blank page margins
    have much lower per-row texture than the fabric body. We measure per-row
    standard deviation as a texture proxy, smooth it to tolerate the thin text
    line inside a bar, and trim edge rows whose texture stays well below the
    region body's. Genuinely plain fabrics have a uniformly low profile, so the
    body threshold is never crossed and nothing is trimmed.

    Returns:
        (top_trim, bottom_trim) in pixels.
    """
    h, w = gray_crop.shape[:2]
    if h < 30 or w < 10:
        return 0, 0

    row_std = gray_crop.std(axis=1).astype(np.float32)

    # Smooth so a sparse text line inside a solid bar doesn't break the band.
    k = max(1, h // 50)
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / k
        row_std = np.convolve(row_std, kernel, mode="same")

    # Body texture = median of the central rows (immune to edge bars).
    lo, hi = int(h * 0.30), int(h * 0.70)
    body = float(np.median(row_std[lo:hi]))
    if body < min_body_std:
        return 0, 0  # essentially uniform region (plain fabric) — leave alone

    thresh = body * ratio
    max_band = int(h * max_band_frac)

    top = 0
    while top < max_band and row_std[top] < thresh:
        top += 1
    if top < min_band_px:
        top = 0

    bottom = 0
    while bottom < max_band and row_std[h - 1 - bottom] < thresh:
        bottom += 1
    if bottom < min_band_px:
        bottom = 0

    # Safety: never collapse the region.
    if top + bottom >= h - min_band_px:
        return 0, 0

    return top, bottom


def _iou(box1, box2):
    """Compute intersection-over-union of two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int]]:
    """
    Iteratively merge boxes that overlap above the IoU threshold.
    """
    if not boxes:
        return boxes

    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i in range(len(merged)):
            if i in used:
                continue
            current = merged[i]

            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                if _iou(current, merged[j]) > iou_threshold:
                    x1, y1, w1, h1 = current
                    x2, y2, w2, h2 = merged[j]
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    current = (nx, ny, nw, nh)
                    used.add(j)
                    changed = True

            new_merged.append(current)
            used.add(i)

        merged = new_merged

    return merged

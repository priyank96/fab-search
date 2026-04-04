"""
fab-search — Configuration constants and environment settings.

All external service credentials are loaded from environment variables.
Processing constants (tiling, segmentation) are kept as module-level defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── External services (env vars) ──────────────────────────────

# Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "fab-search")

# HuggingFace Space (Gradio) for CLIP embeddings
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "knayirp/fab-search")

# Cloudflare R2 storage for page images
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "fab-search")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")  # e.g. https://pub-xxx.r2.dev

# ── CLIP model (must match what the HF Space serves) ──────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512  # ViT-B-32 output dimension

# ── PDF rendering ─────────────────────────────────────────────
DPI = 200

# ── Tiling parameters ────────────────────────────────────────
TILE_SIZE = 1280
STRIDE = 640
BLANK_THRESHOLD = 0.90
WHITE_VALUE = 240

# ── Contour-based segmentation ────────────────────────────────
MIN_REGION_AREA_FRAC = 0.005
MAX_REGION_AREA_FRAC = 0.80
MORPH_KERNEL_SIZE = 9
MERGE_IOU_THRESHOLD = 0.30
REGION_PADDING = 10
TILE_FALLBACK_MIN_REGIONS = 1


# ── Multi-pattern detection ───────────────────────────────────
# When a tile's 4 quadrants have pairwise cosine similarity below
# this threshold, the tile is split into 4 separate vectors.
SIMILARITY_THRESHOLD = 0.75

# ── Search ────────────────────────────────────────────────────
TOP_K = 5

# ── Re-ranking ────────────────────────────────────────────────
# Number of raw tile matches to fetch from Pinecone before re-ranking.
RERANK_FETCH_K = 200
# How many top candidates get patch-level feature comparison (~1-2s).
RERANK_PATCH_CANDIDATES = 30
# Feature weights (must sum to 1.0).
W_EMBEDDING = 0.35       # Best tile CLIP cosine similarity
W_COLOR = 0.20           # HSV histogram correlation
W_TEXTURE = 0.20         # LBP texture histogram similarity
W_EDGE = 0.10            # Canny edge density similarity
W_DOMINANT_COLOR = 0.15  # Top-3 dominant colour match (LAB k-means)

# ── Local paths (for indexing CLI only) ───────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGS_DIR = os.path.join(PROJECT_ROOT, "catalogs")
PAGES_DIR = os.path.join(PROJECT_ROOT, "pages")

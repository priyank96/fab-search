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

# Vercel Blob storage for page images
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "")

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
MORPH_KERNEL_SIZE = 15
MERGE_IOU_THRESHOLD = 0.30
REGION_PADDING = 10
TILE_FALLBACK_MIN_REGIONS = 1

# ── Search ────────────────────────────────────────────────────
TOP_K = 5

# ── Local paths (for indexing CLI only) ───────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGS_DIR = os.path.join(PROJECT_ROOT, "catalogs")
PAGES_DIR = os.path.join(PROJECT_ROOT, "pages")

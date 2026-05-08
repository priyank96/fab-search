"""
Shared training configuration. Reuses model identity from src.config so
the fine-tuned weights are guaranteed loadable by the production
embedding path.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    EMBEDDING_DIM,
)

# ── Paths ─────────────────────────────────────────────────────
TRAIN_ROOT = Path(__file__).resolve().parent
CACHE_DIR = TRAIN_ROOT / "cache"
PAGE_CACHE_DIR = CACHE_DIR / "pages"
MANIFEST_PATH = CACHE_DIR / "manifest.parquet"
CHECKPOINT_DIR = TRAIN_ROOT / "checkpoints"
CONTACT_SHEET_DIR = TRAIN_ROOT / "contact_sheets"

for _p in (CACHE_DIR, PAGE_CACHE_DIR, CHECKPOINT_DIR, CONTACT_SHEET_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── Model identity (must match production HF Space) ───────────
MODEL_NAME = CLIP_MODEL_NAME           # "ViT-B-32"
PRETRAINED = CLIP_PRETRAINED           # "laion2b_s34b_b79k"
EMB_DIM = EMBEDDING_DIM                # 512

# ── Dataset construction ──────────────────────────────────────
# Catalogs held out for validation (split by pdf_name). Override via
# TRAIN_VAL_CATALOGS env (comma-separated) once we know the full list.
VAL_CATALOGS = [
    c.strip()
    for c in os.environ.get("TRAIN_VAL_CATALOGS", "").split(",")
    if c.strip()
]
# If unset, build_dataset.py will pick the 3 smallest catalogs by patch
# count so the train set isn't starved.
DEFAULT_NUM_VAL_CATALOGS = 3

REGION_PADDING_PX = 10  # mirrors src.segment

# ── Augmentation (pattern-safe; see plan Phase 2) ─────────────
INPUT_SIZE = 224
MIN_CROP_SHORT_SIDE = 384       # px floor before final resize
CROP_SCALE = (0.8, 1.0)
CROP_RATIO = (0.85, 1.18)
PERSPECTIVE_DISTORTION = (0.15, 0.30)
TILT_DEGREES = 3.0
BLUR_PROB = 0.30
JPEG_QUALITY = (50, 85)
BRIGHTNESS = 0.30
CONTRAST = 0.30
CLAHE_PROB = 0.20

# OpenCLIP normalisation (OPENAI_DATASET_MEAN/STD)
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)

# ── Training ──────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))
NUM_WORKERS = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "10"))
WARMUP_STEPS = 500
LR_LORA = 1e-4          # LoRA + last block
LR_PROJ = 5e-5          # visual.proj
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 2  # epochs without val Recall@5 improvement

# Hard-negative sampling
HARD_NEG_FRAC = 0.5     # fraction of batch biased to share pdf_name

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")

# ── Logging ───────────────────────────────────────────────────
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "fab-search-finetune")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # set to "offline" or "disabled"

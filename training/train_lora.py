"""
Contrastive fine-tuning loop for FabSearch.

InfoNCE on L2-normalised embeddings, two augmented views per patch.
LoRA + last-block + visual.proj only — see training.model.

Run:
    python -m training.train_lora
"""
from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GRAD_CLIP,
    MANIFEST_PATH,
    NUM_WORKERS,
    SEED,
    WANDB_MODE,
    WANDB_PROJECT,
    WARMUP_STEPS,
)
from training.dataset import HardNegativeBatchSampler, PatchDataset
from training.eval import run_proxy_eval
from training.model import build_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def info_nce(z1: torch.Tensor, z2: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE. z1, z2 are L2-normalised, shape (N, D)."""
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * (z1 @ z2.T)
    targets = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def cosine_warmup_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Missing {MANIFEST_PATH}. Run build_dataset.py + flag_single_pattern.py first.")
    df = pd.read_parquet(MANIFEST_PATH)
    if "single_pattern" not in df.columns or df["single_pattern"].isna().any():
        log.warning("single_pattern flag missing; quadrant pairs disabled. Run flag_single_pattern.py for stronger positives.")
        df["single_pattern"] = False
    df["single_pattern"] = df["single_pattern"].fillna(False).astype(bool)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    log.info("Train: %d  Val: %d  (single_pattern train: %d)",
             len(train_df), len(val_df), int(train_df["single_pattern"].sum()))

    train_ds = PatchDataset(train_df, train=True, use_quadrant_pairs=True)

    steps_per_epoch = max(1, len(train_df) // BATCH_SIZE)
    sampler = HardNegativeBatchSampler(
        train_df, batch_size=BATCH_SIZE, num_batches=steps_per_epoch, seed=SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )

    encoder, param_groups = build_model(device)
    optim = torch.optim.AdamW(param_groups)
    # Snapshot the configured LR per group so the cosine schedule has a
    # stable base to scale from each step.
    base_lrs = [g["lr"] for g in optim.param_groups]
    total_steps = steps_per_epoch * EPOCHS

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ── wandb (optional) ──────────────────────────────────────
    try:
        import wandb
        wandb.init(project=WANDB_PROJECT, mode=WANDB_MODE, config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model": "open_clip ViT-B-32 laion2b_s34b_b79k",
            "strategy": "LoRA + last block + visual.proj",
        })
    except Exception:
        wandb = None
        log.warning("wandb unavailable; logging to stdout only")

    # ── Baseline eval before any training step ────────────────
    log.info("Running pretrained-baseline proxy eval...")
    base_metrics = run_proxy_eval(encoder, val_df, device)
    log.info("Baseline metrics: %s", base_metrics)
    if wandb:
        wandb.log({f"baseline/{k}": v for k, v in base_metrics.items()})

    best_r5 = base_metrics.get("recall@5", 0.0)
    best_path = CHECKPOINT_DIR / "best.pt"
    patience = 0
    global_step = 0

    for epoch in range(EPOCHS):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            v1 = batch["view1"].to(device, non_blocking=True)
            v2 = batch["view2"].to(device, non_blocking=True)

            # LR schedule
            lr_mult = cosine_warmup_lr(global_step, total_steps, WARMUP_STEPS)
            for g, base in zip(optim.param_groups, base_lrs):
                g["lr"] = base * lr_mult

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                z1 = encoder(v1)
                z2 = encoder(v2)
                loss = info_nce(z1, z2, encoder.logit_scale)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optim.param_groups for p in g["params"]], GRAD_CLIP
                )
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optim.param_groups for p in g["params"]], GRAD_CLIP
                )
                optim.step()

            with torch.no_grad():
                encoder.logit_scale.clamp_(0.0, math.log(100.0))

            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", logit_scale=f"{encoder.logit_scale.exp().item():.2f}")
            if wandb and global_step % 25 == 0:
                wandb.log({
                    "train/loss": float(loss.item()),
                    "train/logit_scale": float(encoder.logit_scale.exp().item()),
                    "train/lr_mult": lr_mult,
                    "step": global_step,
                })

        avg_loss = epoch_loss / max(1, n_batches)
        log.info("Epoch %d: avg loss %.4f", epoch, avg_loss)

        # Eval
        encoder.eval()
        val_metrics = run_proxy_eval(encoder, val_df, device)
        log.info("Epoch %d val: %s", epoch, val_metrics)
        if wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})

        r5 = val_metrics.get("recall@5", 0.0)
        if r5 > best_r5 + 1e-4:
            best_r5 = r5
            patience = 0
            torch.save({
                "encoder_state": encoder.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, best_path)
            log.info("Saved new best to %s (Recall@5=%.4f)", best_path, r5)
        else:
            patience += 1
            log.info("No improvement (%d/%d).", patience, EARLY_STOP_PATIENCE)
            if patience >= EARLY_STOP_PATIENCE:
                log.info("Early stopping.")
                break

    log.info("Training done. Best Recall@5: %.4f -> %s", best_r5, best_path)


if __name__ == "__main__":
    main()

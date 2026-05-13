"""
Build the trainable image-only OpenCLIP model with LoRA + last-block
unfreeze + visual.proj training.

Returns the inner image tower (so callers don't accidentally use the
text branch) plus a list of parameter groups for the optimiser.
"""
from __future__ import annotations

import logging
from typing import Any

import open_clip
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from training.config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
    LR_LORA,
    LR_PROJ,
    MODEL_NAME,
    PRETRAINED,
    WEIGHT_DECAY,
)

log = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """Wraps OpenCLIP's visual tower; output is L2-normalised."""

    def __init__(self, visual: nn.Module, logit_scale: nn.Parameter):
        super().__init__()
        self.visual = visual
        # Re-use CLIP's logit scale; learnable.
        self.logit_scale = logit_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.visual(x)
        return nn.functional.normalize(feat, dim=-1)


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_last_block(visual: nn.Module) -> list[nn.Parameter]:
    """Find the last residual attention block and unfreeze its non-LoRA params.

    open_clip's ViT visual: visual.transformer.resblocks[-1]. The block
    contains the q/k/v/out_proj linears that PEFT has already wrapped
    with LoRA adapters; we must NOT re-collect those adapter params or
    the optimiser will see them in two param groups.
    """
    params: list[nn.Parameter] = []
    blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
    if blocks is None:
        log.warning("Could not locate transformer.resblocks; skipping last-block unfreeze")
        return params
    last = blocks[-1]
    for name, p in last.named_parameters():
        if "lora_" in name:
            # Already trainable via PEFT; do not duplicate.
            continue
        p.requires_grad = True
        params.append(p)
    log.info("Unfroze last block (excl. LoRA): %d params", sum(p.numel() for p in params))
    return params


def _unfreeze_proj(visual: nn.Module) -> list[nn.Parameter]:
    """Unfreeze visual.proj (the final 512-d projection)."""
    proj = getattr(visual, "proj", None)
    if proj is None or not isinstance(proj, torch.nn.Parameter):
        log.warning("visual.proj not found or not a Parameter; skipping")
        return []
    proj.requires_grad = True
    log.info("Unfroze visual.proj: %d params", proj.numel())
    return [proj]


def build_model(
    device: torch.device,
    init_visual_from: str | None = None,
) -> tuple[ImageEncoder, list[dict[str, Any]]]:
    """Returns (encoder, optimizer_param_groups).

    If ``init_visual_from`` is given, it must be a path to an exported
    checkpoint (``training.export``) whose ``model_state`` will be loaded
    into the fresh open_clip model BEFORE LoRA wrapping. This lets us
    continue training from a merged finetune.
    """
    log.info("Loading open_clip %s / %s ...", MODEL_NAME, PRETRAINED)
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)

    if init_visual_from:
        log.info("Initialising from merged checkpoint: %s", init_visual_from)
        ckpt = torch.load(init_visual_from, map_location="cpu")
        merged = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
        missing, unexpected = model.load_state_dict(merged, strict=False)
        log.info("Loaded merged weights (missing=%d unexpected=%d)", len(missing), len(unexpected))
        if unexpected:
            log.warning("Unexpected keys (first 3): %s", unexpected[:3])

    # Image-only: drop text components to free memory and prevent accidental use.
    for attr in ("transformer", "token_embedding", "positional_embedding", "ln_final", "text_projection"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:
                pass

    visual = model.visual
    logit_scale = model.logit_scale

    # Freeze everything first.
    _freeze_all(visual)
    logit_scale.requires_grad = True  # learnable temperature

    # Apply LoRA to attention projections in the visual transformer.
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=list(LORA_TARGETS),
    )
    visual = get_peft_model(visual, lora_cfg)
    n_trainable = sum(p.numel() for p in visual.parameters() if p.requires_grad)
    log.info("LoRA trainable params: %d", n_trainable)

    # Reach into the wrapped model for last-block + proj unfreezes.
    inner_visual = visual.base_model.model if hasattr(visual, "base_model") else visual
    last_block_params = _unfreeze_last_block(inner_visual)
    proj_params = _unfreeze_proj(inner_visual)

    encoder = ImageEncoder(visual=visual, logit_scale=logit_scale).to(device)

    # Build param groups with separate LRs.
    lora_params = [
        p for n, p in encoder.visual.named_parameters()
        if p.requires_grad and "lora_" in n
    ]
    # Avoid double-counting: last-block + proj are NOT lora_ params.
    other_params = last_block_params + proj_params + [logit_scale]
    other_params = [p for p in other_params if p.requires_grad]

    param_groups = [
        {"params": lora_params, "lr": LR_LORA, "weight_decay": WEIGHT_DECAY, "name": "lora"},
        {"params": other_params, "lr": LR_PROJ, "weight_decay": WEIGHT_DECAY, "name": "proj+last"},
    ]
    total = sum(p.numel() for g in param_groups for p in g["params"])
    log.info("Total trainable params: %d", total)
    return encoder, param_groups

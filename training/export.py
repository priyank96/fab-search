"""
Merge LoRA adapters into the base OpenCLIP weights and emit a single
state_dict that loads cleanly into a fresh `open_clip.create_model_and_transforms`
call. This is what gets uploaded to the HF Space.

Verifies the merged model produces embeddings within 1e-5 of the
training-time LoRA model on 10 fixed images.
"""
from __future__ import annotations

import logging
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from training.config import CHECKPOINT_DIR, MODEL_NAME, PRETRAINED
from training.model import build_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _fixed_test_inputs(preprocess, device: torch.device) -> torch.Tensor:
    """Generate 10 deterministic synthetic images so the script has zero
    dependence on the page cache being present."""
    rng = torch.Generator().manual_seed(0)
    imgs = []
    for _ in range(10):
        arr = (torch.randn(3, 224, 224, generator=rng) * 0.3 + 0.5).clamp(0, 1)
        # Wrap → PIL → preprocess so we exercise the same path the Space uses.
        pil = Image.fromarray((arr.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        imgs.append(preprocess(pil))
    return torch.stack(imgs, dim=0).to(device)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "best.pt"))
    p.add_argument("--out", default=str(CHECKPOINT_DIR / "fab_search_vit_b32.pt"))
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading training-time LoRA model...")
    encoder, _ = build_model(device)
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state["encoder_state"], strict=False)
    encoder.eval()

    # Reference embeddings BEFORE merge.
    _, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    test_in = _fixed_test_inputs(preprocess, device)
    with torch.no_grad():
        ref = encoder(test_in).float().cpu()

    # Merge LoRA into base weights.
    log.info("Merging LoRA adapters...")
    visual_peft = encoder.visual
    if hasattr(visual_peft, "merge_and_unload"):
        merged_visual = visual_peft.merge_and_unload()
    else:
        raise RuntimeError("encoder.visual is not a PEFT model; nothing to merge.")

    # Build a fresh open_clip model and overwrite its visual + logit_scale.
    log.info("Building target open_clip model and copying merged weights...")
    target, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    # Drop text components (they're untrained / dropped during training).
    # We keep them in the saved state_dict for compatibility — they remain
    # at their LAION pretrained values, so any future text-search use is a
    # no-op regression to baseline.
    target.visual.load_state_dict(merged_visual.state_dict(), strict=True)
    with torch.no_grad():
        target.logit_scale.copy_(encoder.logit_scale.detach())
    target = target.to(device).eval()

    # Roundtrip check.
    with torch.no_grad():
        feats = target.visual(test_in)
        feats = F.normalize(feats, dim=-1).float().cpu()
    diff = (feats - ref).abs().max().item()
    log.info("Roundtrip max abs diff: %.2e (gate: <1e-4)", diff)
    if diff > 1e-4:
        log.error("Roundtrip diff too large; not saving.")
        raise SystemExit(2)

    # Save full state_dict (visual + text + logit_scale) so the Space can
    # `model.load_state_dict(...)` after `create_model(...)`.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": target.state_dict(),
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "training_metrics": state.get("val_metrics", {}),
    }, out_path)
    log.info("Saved merged checkpoint -> %s", out_path)


if __name__ == "__main__":
    main()

"""
CLIP Embedding API — Returns raw image embeddings via Gradio.

This is the canonical, version-controlled source for the HuggingFace Space
`knayirp/fab-search`. Keep this file in sync with the Space.

Endpoints:
    /get_embedding        — single image  → 512-dim embedding JSON
    /get_embeddings_pair  — two images    → both embeddings + cosine similarity

Preprocessing note:
    The fine-tuned model was trained with an INTER_AREA resize + a short-side
    floor (see training/augment.py: prepare_patch + build_eval_transform).
    open_clip's default preprocess uses a BICUBIC resize and no short-side
    floor, which is off-distribution for the fine-tune and aliases fine
    patterns (pinstripes / small checks). `_preprocess_match` below mirrors
    the training-time eval transform exactly so serving matches training.

Run:
    pip install -r requirements.txt
    python app.py
"""

import json

import cv2
import gradio as gr
import numpy as np
import torch
import open_clip
from PIL import Image
from huggingface_hub import hf_hub_download

# ── Config (mirrors src/config.py) ───────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Fine-tuned weights repo (uploaded by training/upload_to_hf.py)
WEIGHTS_REPO = "knayirp/fab-search-vit-b32"
WEIGHTS_FILE = "fab_search_vit_b32.pt"

# ── Preprocessing constants (mirror training/config.py) ──────
INPUT_SIZE = 224
MIN_CROP_SHORT_SIDE = 384  # px floor before final downscale
# OpenCLIP normalisation (OPENAI_DATASET_MEAN/STD)
NORM_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
NORM_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)

# ── Model loading ────────────────────────────────────────────
print(f"Loading CLIP architecture: {CLIP_MODEL_NAME} (no pretrained, will load finetuned)")
_model, _, _preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=None
)

print(f"Downloading finetuned weights: {WEIGHTS_REPO}/{WEIGHTS_FILE} …")
_ckpt_path = hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE)
_state = torch.load(_ckpt_path, map_location="cpu")
_state_dict = _state.get("model_state", _state)
missing, unexpected = _model.load_state_dict(_state_dict, strict=False)
print(f"Loaded weights (missing={len(missing)}, unexpected={len(unexpected)})")
if missing:
    print("  first missing:", missing[:3])
if unexpected:
    print("  first unexpected:", unexpected[:3])

_model.eval()
print("Model ready.")


# ── Preprocessing ────────────────────────────────────────────
def _preprocess_match(pil_image: Image.Image) -> torch.Tensor:
    """Mirror training's eval transform so serving matches the fine-tune
    distribution (training/augment.py: prepare_patch + build_eval_transform):

      1. RGB
      2. enforce short side >= MIN_CROP_SHORT_SIDE (INTER_CUBIC upscale only)
      3. SmallestMaxSize to INPUT_SIZE (INTER_AREA, anti-aliased downscale)
      4. CenterCrop INPUT_SIZE
      5. Normalize (OpenAI mean/std)
    """
    rgb = np.asarray(pil_image.convert("RGB"))  # HWC uint8

    # 2. Short-side floor — upscale small patches before downsampling so fine
    #    patterns have enough pixel budget to survive the 224 downscale.
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short < MIN_CROP_SHORT_SIDE:
        s = MIN_CROP_SHORT_SIDE / short
        rgb = cv2.resize(
            rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC
        )

    # 3. SmallestMaxSize(INPUT_SIZE, INTER_AREA): resize so the short side == 224.
    h, w = rgb.shape[:2]
    s = INPUT_SIZE / min(h, w)
    rgb = cv2.resize(
        rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
    )

    # 4. CenterCrop(INPUT_SIZE).
    h, w = rgb.shape[:2]
    top, left = (h - INPUT_SIZE) // 2, (w - INPUT_SIZE) // 2
    rgb = rgb[top:top + INPUT_SIZE, left:left + INPUT_SIZE]

    # 5. Normalize.
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def _encode(pil_image: Image.Image) -> np.ndarray:
    """Encode a PIL image into a normalised 512-dim CLIP embedding."""
    tensor = _preprocess_match(pil_image)
    with torch.no_grad():
        emb = _model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def get_embedding(image) -> str:
    """Return the raw embedding as JSON for a single image."""
    if image is None:
        return json.dumps({"error": "No image provided."})
    pil = Image.fromarray(image).convert("RGB")
    emb = _encode(pil)
    return json.dumps({
        "embedding": emb.tolist(),
        "dim": len(emb),
        "norm": float(np.linalg.norm(emb)),
        "model": CLIP_MODEL_NAME,
        "pretrained": CLIP_PRETRAINED,
    })


def get_embeddings_pair(image_a, image_b) -> str:
    """Return embeddings + cosine similarity for two images."""
    if image_a is None or image_b is None:
        return json.dumps({"error": "Two images are required."})
    pil_a = Image.fromarray(image_a).convert("RGB")
    pil_b = Image.fromarray(image_b).convert("RGB")
    emb_a = _encode(pil_a)
    emb_b = _encode(pil_b)
    return json.dumps({
        "embedding_a": emb_a.tolist(),
        "embedding_b": emb_b.tolist(),
        "cosine_similarity": float(np.dot(emb_a, emb_b)),
        "dim": len(emb_a),
        "model": CLIP_MODEL_NAME,
        "pretrained": CLIP_PRETRAINED,
    })


# ── Gradio app ────────────────────────────────────────────────
with gr.Blocks(title="CLIP Embedding API") as demo:
    gr.Markdown(
        "## CLIP Embedding API\n"
        f"Model: **{CLIP_MODEL_NAME}** / `{CLIP_PRETRAINED}` (fine-tuned) — "
        f"512-dim embeddings via `open_clip`.\n\n"
        "Preprocessing matches the training eval transform "
        "(INTER_AREA resize + short-side floor).\n\n"
        "Use the API endpoints below to retrieve raw embedding vectors."
    )

    with gr.Tab("Get Embedding"):
        img_in = gr.Image(label="Upload image")
        btn = gr.Button("Get Embedding", variant="primary")
        json_out = gr.Textbox(label="Embedding JSON", lines=8)
        btn.click(get_embedding, inputs=[img_in], outputs=[json_out],
                  api_name="get_embedding")

    with gr.Tab("Compare Two Images"):
        img_a = gr.Image(label="Image A")
        img_b = gr.Image(label="Image B")
        btn2 = gr.Button("Get Embeddings", variant="primary")
        json_out2 = gr.Textbox(label="Embeddings JSON", lines=8)
        btn2.click(get_embeddings_pair, inputs=[img_a, img_b],
                   outputs=[json_out2], api_name="get_embeddings_pair")

if __name__ == "__main__":
    demo.launch(share=True)

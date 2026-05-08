# `training/` — FabSearch fine-tuning

Isolated workspace for fine-tuning the OpenCLIP `ViT-B-32 / laion2b_s34b_b79k`
vision tower used by the production HuggingFace Space. **Not imported by the
deployed FastAPI app.**

## Run order

Set up:

```bash
pip install -r training/requirements-train.txt
# .env must have PINECONE_API_KEY, PINECONE_INDEX_NAME (already used by src/)
```

1. **Build the manifest** — pulls all vector metadata from Pinecone, downloads each unique page JPEG once into `training/cache/pages/`, and writes `training/cache/manifest.parquet`. CPU only.
   ```bash
   python -m training.build_dataset
   ```
2. **Flag single-pattern tiles** — runs the pretrained CLIP quadrant probe ([src/tiler.py](src/tiler.py#L65-L102) logic) and adds the `single_pattern` boolean column to the manifest. Needs torch + GPU recommended.
   ```bash
   python -m training.flag_single_pattern
   ```
3. **Sanity-check augmentations** — render contact sheets BEFORE spending GPU. Verify pattern detail survives in both views.
   ```bash
   python -m training.contact_sheet --kind aug   --n 16
   python -m training.contact_sheet --kind quad  --n 16
   python -m training.contact_sheet --kind fine  --n 50    # plan §"Verification" #1
   ```
   Output goes to `training/contact_sheets/`. **If pinstripes look like solid grey or aliased Moiré, tighten the augmentation knobs in [training/config.py](training/config.py) before training.**
4. *(Optional)* **Hand-curate the pattern probe CSV** — to get the per-bucket Recall@5 deployment gate, label ~40+ val patches by ID into buckets:
   ```text
   training/pattern_buckets.csv

   id,bucket
   <vector-id>,fine_stripe
   <vector-id>,small_check
   ...
   ```
5. **Train** — InfoNCE on LoRA + last block + visual.proj. Logs to wandb if available.
   ```bash
   python -m training.train_lora
   ```
   Best checkpoint saved to `training/checkpoints/best.pt`.
6. **Evaluate the best checkpoint** standalone:
   ```bash
   python -m training.eval --ckpt training/checkpoints/best.pt
   ```
7. **Export merged weights** for the HF Space:
   ```bash
   python -m training.export --ckpt training/checkpoints/best.pt
   ```
   Produces `training/checkpoints/fab_search_vit_b32.pt` containing
   `{"model_state": ..., "model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", ...}`. Roundtrip-verified to within 1e-4 of training-time embeddings.

## Deployment (after export)

On the HF Space (`knayirp/fab-search`):

```python
import open_clip, torch
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
state = torch.load("fab_search_vit_b32.pt", map_location="cpu")
model.load_state_dict(state["model_state"], strict=True)
```

Keep the `/get_embedding` Gradio endpoint signature unchanged so [src/embedding_client.py](src/embedding_client.py) continues to work. Bump a `MODEL_VERSION` env var in the Space so production can verify the new weights are live before re-indexing.

After verification:

```bash
# Drop & recreate the Pinecone index (dim unchanged, schema unchanged).
python -m src.index_cli rebuild
```

## Decisions baked in

- **Image-only contrastive** — text tower dropped; output stays 512-d.
- **LoRA (r=16) on q/k/v/out_proj + last transformer block + visual.proj.**
- **Pattern-safe augmentations** — anti-aliased downscale (`INTER_AREA`), no h-flip, no general rotation, perspective ≤ 0.30, JPEG ≥ Q50, light blur.
- **Hard negatives** — `HardNegativeBatchSampler` puts ~50% of each batch from same-catalog confusables.
- **Sqrt-frequency catalog weighting** so large catalogs don't dominate.
- **Splits by `pdf_name`** — 3 catalogs held out for val (override via `TRAIN_VAL_CATALOGS`).

## Env knobs

| Var | Default | Purpose |
|---|---|---|
| `TRAIN_VAL_CATALOGS` | (auto: 3 smallest ≥10 patches) | Comma-separated list of catalog filenames for val |
| `TRAIN_BATCH_SIZE` | 128 | 256 on A100, 128 on T4 |
| `TRAIN_EPOCHS` | 10 | Early stops on val Recall@5 plateau |
| `TRAIN_NUM_WORKERS` | 4 | DataLoader workers |
| `WANDB_PROJECT` | `fab-search-finetune` | |
| `WANDB_MODE` | `online` | Set to `disabled` for no logging |

## Deployment gates (from plan)

1. Val **Recall@5 ≥ +5 pp** over pretrained baseline.
2. Per-bucket Recall@5 does not regress on `fine_stripe / wide_stripe / small_check / large_check`.
3. Quadrant intra/inter cosine gap widens.
4. Export roundtrip max abs diff < 1e-4 on 10 fixed images.

# fab-search — Specification Document

## 1. Overview

fab-search is a serverless web application that matches phone photos of fabric patterns against textile catalog PDFs. A user uploads a photo from their phone, and the system returns the matching catalog name, page number, and a highlighted image of the match.

The application is designed mobile-first (accessible from any phone browser without installation), powered by CLIP-based visual similarity search, and deployed entirely on managed cloud services — no local servers or native dependencies required.

| Aspect | Detail |
|---|---|
| Deployment | Vercel serverless (Python runtime) |
| Frontend | Mobile-first HTML/CSS/JS (vanilla, no framework) |
| Embedding model | CLIP ViT-B-32 served via HuggingFace Space |
| Vector database | Pinecone (512-dim, cosine similarity) |
| Page image storage | Vercel Blob (CDN-backed) |
| PDF rendering | PyMuPDF (pure pip, zero system dependencies) |
| Indexing | Standalone offline CLI |

---

## 2. Problem Statement

- A cloth wholesaler receives phone photos of fabric patterns from customers.
- They need to identify which catalog(s) contain that pattern and on which page.
- Input images are phone camera photos — they contain shadows, uneven lighting, perspective distortion, and background objects (not just the fabric).
- A single PDF catalog page may contain multiple fabric patterns.
- The match is a **visual similarity** match between a noisy real-world photo and a clean catalog scan.
- The tool must be accessible from mobile phones (the primary user device) without any app installation — a responsive website that works on any browser.

---

## 3. Approach

**CLIP-based Semantic Image Matching** with **Pinecone** for managed vector search and **Vercel** for serverless hosting.

### Why CLIP?
- CLIP embeddings are robust to the domain gap between phone photos and clean catalog scans.
- Zero-shot — no training data or fine-tuning required.
- Handles lighting variation, shadows, and perspective distortion well.

### Why a remote HuggingFace Space for inference?
- The CLIP model (PyTorch + open_clip) weighs ~2 GB — far exceeding Vercel's 250 MB function size limit.
- A dedicated HF Space (`knayirp/fab-search`) serves the ViT-B-32 model via a Gradio API.
- This cleanly decouples the heavy ML inference from the lightweight API layer.

### Why Pinecone?
- Vercel serverless functions have no persistent filesystem — file-based indexes cannot be stored.
- Pinecone is a managed vector database with built-in metadata filtering.
- Tile metadata (PDF name, page number, coordinates, page image URL) is stored directly on each vector — no separate database needed.

### Why PyMuPDF?
- System-level PDF renderers (like Poppler) cannot be installed on Vercel's runtime.
- PyMuPDF bundles MuPDF binaries inside the pip wheel (~30 MB) — zero system dependencies, installs with `pip install pymupdf`.

### Why Vercel Blob?
- Serverless functions have no persistent disk — rendered page images need cloud storage.
- Vercel Blob provides simple PUT-based uploads and CDN-backed public URLs.
- Page images are uploaded once during indexing; search results return blob URLs directly.

---

## 4. Architecture

### 4.1 System Diagram

```
                         ┌──────────────────────────────────┐
                         │         Vercel (Serverless)       │
                         │                                   │
  Mobile/Desktop  ──────►│  FastAPI  ──►  Pinecone (query)   │
  Browser                │     │                             │
  (HTML/CSS/JS)   ◄──────│     ▼                             │
                         │  HF Space  (CLIP embedding)       │
                         │                                   │
                         │  Vercel Blob (page image CDN)     │
                         └──────────────────────────────────┘

                         ┌──────────────────────────────────┐
                         │      Indexing CLI (offline)        │
                         │                                   │
  PDF catalogs  ────────►│  PyMuPDF (render pages)           │
                         │  OpenCV  (segment regions)        │
                         │  HF Space (CLIP embeddings)       │
                         │  Pinecone (upsert vectors)        │
                         │  Vercel Blob (upload page PNGs)   │
                         └──────────────────────────────────┘
```

### 4.2 Indexing Pipeline (Offline — CLI)

Indexing runs as a standalone CLI (`python -m src.index_cli`) on a local machine or CI worker. It cannot run inside Vercel serverless functions because PDF processing takes minutes per catalog, exceeding the serverless timeout (10–60 s).

```
PDF file
  → Render each page to image (PyMuPDF, 200 DPI)
  → Upload page PNG to Vercel Blob → get public URL
  → For each page:
      → Compute full-page CLIP embedding (via HF Space)
      → Store full-page vector in Pinecone (always)
      → Detect pattern regions via contour segmentation (OpenCV):
          → Grayscale → blur → adaptive threshold → morphological close → find contours
          → Filter by area (0.5%–80% of page) and aspect ratio (<8:1)
          → Merge overlapping boxes (IoU > 0.30)
      → If regions detected ≥ TILE_FALLBACK_MIN_REGIONS:
          → Crop each region, encode with CLIP, store in Pinecone
      → Else (fallback):
          → Generate sliding window tiles (1280×1280 px, 50% overlap)
          → Filter blank tiles (≥90% near-white pixels)
          → Encode each tile with CLIP, store in Pinecone
  → Pinecone metadata per vector:
      pdf_name, page_number, tile_x, tile_y, tile_w, tile_h, page_image (blob URL)
```

### 4.3 Search Pipeline (Online — Vercel Function)

```
User uploads photo (mobile camera or file picker)
  → POST /api/search  (image file + top_k parameter)
  → Encode query image via HF Space → 512-dim CLIP vector
  → Query Pinecone (cosine similarity, over-fetch ×20)
  → Deduplicate results by (pdf_name, page_number)
  → Return top-K unique pages with:
      rank, score, pdf_name, page_number,
      tile_x/y/w/h (highlight coordinates), page_image (blob URL)
  → Frontend renders results with red highlight overlay on matched region
```

---

## 5. Segmentation Strategy

Each catalog page may contain multiple fabric patterns. The indexing pipeline must isolate individual patterns so each gets its own embedding vector.

### 5.1 Primary: Contour-Based Detection

Adaptive contour detection locates pattern regions automatically, producing bounding boxes that match actual pattern boundaries.

**Pipeline:**
1. Convert page to grayscale, apply Gaussian blur (5×5)
2. Adaptive threshold (GAUSSIAN_C, BINARY_INV) — handles varying backgrounds
3. Morphological close (15×15 kernel, 2 iterations) + dilate (7×7)
4. Find external contours → bounding rects
5. Filter: area in [0.5%, 80%] of page, aspect ratio ≤ 8:1
6. Add 10 px padding, merge overlapping boxes (IoU > 0.30)
7. Sort top-to-bottom, left-to-right

| Parameter | Value | Config key | Effect |
|---|---|---|---|
| Min region area | 0.5% of page | `MIN_REGION_AREA_FRAC` | Smaller → keep tinier regions |
| Max region area | 80% of page | `MAX_REGION_AREA_FRAC` | Larger → allow bigger regions |
| Morph kernel | 15 px | `MORPH_KERNEL_SIZE` | Larger → fills bigger gaps |
| Merge IoU | 0.30 | `MERGE_IOU_THRESHOLD` | Lower → more aggressive merging |
| Padding | 10 px | `REGION_PADDING` | Ensures pattern edges aren't clipped |

### 5.2 Fallback: Sliding Window Tiling

When contour detection finds fewer than `TILE_FALLBACK_MIN_REGIONS` regions on a page (e.g., pages with full-bleed photos or models wearing clothes), the system falls back to a fixed grid of overlapping tiles.

| Parameter | Value | Config key |
|---|---|---|
| Tile size | 1280×1280 px | `TILE_SIZE` |
| Stride | 640 px (50% overlap) | `STRIDE` |
| Blank threshold | 0.90 | `BLANK_THRESHOLD` |
| White value | 240 | `WHITE_VALUE` |
| Fallback trigger | < 1 region | `TILE_FALLBACK_MIN_REGIONS` |

A tile is considered blank (and discarded) when ≥ 90% of its pixels have all of R, G, B > 240.

### 5.3 Full-Page Embedding

A full-page CLIP embedding is **always** stored for every page, regardless of whether contour detection or tiling is used. This ensures that even if segmentation fails to isolate a specific pattern, the page can still be matched at the whole-page level.

---

## 6. External Services

### 6.1 HuggingFace Space — `knayirp/fab-search`

Hosts the CLIP ViT-B-32 model (pretrained: laion2b_s34b_b79k) behind a Gradio interface.

**API endpoint used:** `/get_embedding`

```
Input:  image file (PNG/JPG)
Output: JSON string → {"embedding": [512 floats], "shape": [512], "model": "..."}
```

- Called via `gradio_client.Client.predict()`
- Images are saved to a temp file, passed via `handle_file()`
- Supports `HF_TOKEN` env var for private spaces

**Second endpoint (not used in core pipeline):** `/get_embeddings_pair`
- Accepts two images, returns both embeddings + cosine similarity
- Useful for debugging and comparison

### 6.2 Pinecone — `fab-search` Index

| Setting | Value |
|---|---|
| Index name | `fab-search` |
| Dimensions | 512 |
| Metric | cosine |

**Metadata schema per vector:**

| Field | Type | Example |
|---|---|---|
| `pdf_name` | string | `"School Wear Edition 7.pdf"` |
| `page_number` | integer | `3` |
| `tile_x` | integer | `0` |
| `tile_y` | integer | `420` |
| `tile_w` | integer | `800` |
| `tile_h` | integer | `600` |
| `page_image` | string | `"https://....vercel-storage.com/pages/..."` |

**Operations:**
- `upsert` — batch of 100 vectors during indexing
- `query` — cosine similarity search with over-fetch (×20) for page deduplication
- `delete` — by metadata filter (`pdf_name`) for catalog removal
- Catalog listing — zero-vector query with top_k=10000 + metadata aggregation

### 6.3 Vercel Blob Storage

Stores rendered page PNG images, served via CDN-backed public URLs.

**Upload:** `PUT https://blob.vercel-storage.com/{path}` with Bearer token.
**Naming convention:** `pages/{pdf_filename}/{pdf_stem}_page_{NNNN}.png`
**Cache:** `max-age=31536000` (1 year — page images are immutable once rendered).

---

## 7. API Endpoints

All API routes are served by FastAPI on Vercel Python runtime.

### `POST /api/search`

Search the catalog index with an uploaded image.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | UploadFile | required | Query image (JPG/PNG/WebP) |
| `top_k` | int (query) | 5 | Number of unique-page results (1–20) |

**Response:**
```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.8234,
      "pdf_name": "School Wear Edition 7.pdf",
      "page_number": 14,
      "tile_x": 120,
      "tile_y": 340,
      "tile_w": 680,
      "tile_h": 520,
      "page_image": "https://....vercel-storage.com/pages/..."
    }
  ],
  "total": 5
}
```

### `GET /api/catalogs`

List all indexed catalogs.

**Response:**
```json
{
  "catalogs": [
    {"pdf_name": "School Wear Edition 7.pdf", "num_pages": 42, "num_vectors": 186}
  ],
  "total": 1
}
```

### `DELETE /api/catalogs/{catalog_name}`

Remove a catalog and all its vectors from Pinecone.

**Response:**
```json
{"message": "Catalog 'School Wear Edition 7.pdf' removed", "catalog_name": "School Wear Edition 7.pdf"}
```

### `GET /api/health`

Health check. Returns `{"status": "ok"}`.

---

## 8. Indexing CLI

The indexing pipeline runs as a standalone CLI tool on a local machine — not on Vercel.

```bash
# List indexed catalogs
python -m src.index_cli list

# Index a single PDF (or re-index if it already exists)
python -m src.index_cli add catalogs/MyBook.pdf

# Remove a catalog from the index
python -m src.index_cli remove "MyBook.pdf"

# Full rebuild from all PDFs in catalogs/
python -m src.index_cli rebuild
```

Indexing is offline because PDF processing (render → segment → encode → upload → upsert) takes minutes per catalog. Vercel serverless functions timeout at 10 s (hobby) / 60 s (pro), making in-request indexing impossible.

---

## 9. Frontend

Mobile-first single-page application served as static files from `public/`.

### 9.1 Pages

| Page | Route | Purpose |
|---|---|---|
| Search | `/` | Upload image, view results |
| Catalogs | `#catalogs` | View/remove indexed catalogs |

### 9.2 Search UX

1. **Upload:** Tap the upload area → opens camera or file picker. Uses `<input accept="image/*" capture="environment">` for direct phone camera access.
2. **Configure:** Slider for number of results (1–20, default 5).
3. **Search:** Tap button → `POST /api/search` with the image.
4. **Results:** Cards showing the matching page image (loaded from Vercel Blob) with:
   - Red rectangle overlay highlighting the matched tile/region
   - Rank badge and similarity score as a percentage
   - Catalog name and page number

### 9.3 Technology

- Vanilla HTML + CSS + JS (no framework)
- Dark theme, responsive from 320 px up
- CSS custom properties for theming
- Drag-and-drop file upload support
- Canvas overlay for drawing highlight rectangles on results

---

## 10. Project Structure

```
fab-search/
├── main.py                      # FastAPI app entry point (Vercel)
├── vercel.json                  # Vercel routing & build config
├── pyproject.toml               # Project metadata & dependencies
├── requirements.txt             # Pip dependencies
├── .env                         # Secrets (git-ignored)
├── .env.example                 # Template for .env
│
├── api/
│   ├── search.py                # POST /api/search
│   └── catalogs.py              # GET/DELETE /api/catalogs
│
├── src/
│   ├── config.py                # Constants + env var loading
│   ├── embedding_client.py      # HF Space Gradio client
│   ├── pinecone_store.py        # Pinecone vector store wrapper
│   ├── blob_store.py            # Vercel Blob upload client
│   ├── pdf_processor.py         # PDF → images (PyMuPDF)
│   ├── segment.py               # Contour-based region detection
│   ├── tiler.py                 # Sliding window tiling (fallback)
│   └── index_cli.py             # CLI: list/add/remove/rebuild
│
├── public/
│   ├── index.html               # SPA shell
│   ├── style.css                # Mobile-first responsive styles
│   └── app.js                   # Upload, search, results UI
│
├── catalogs/                    # Source PDFs (local, for indexing CLI)
└── pages/                       # Rendered PNGs (local temp, uploaded to Blob)
```

---

## 11. Dependencies

### 11.1 Vercel Runtime (serverless functions)

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `gradio_client` | Call HF Space for CLIP embeddings |
| `pinecone` | Managed vector database client |
| `httpx` | HTTP client (Vercel Blob uploads) |
| `numpy` | Array operations |
| `Pillow` | Image handling |
| `python-dotenv` | Load `.env` file |

### 11.2 Indexing CLI (local machine only)

All of the above, plus:

| Package | Purpose |
|---|---|
| `pymupdf` | PDF → image rendering |
| `opencv-python-headless` | Contour-based segmentation |

### 11.3 External Services

| Service | Purpose | Config env var |
|---|---|---|
| HuggingFace Space | CLIP embedding inference | `HF_SPACE_ID` |
| Pinecone | Vector index + metadata | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` |
| Vercel Blob | Page image CDN storage | `BLOB_READ_WRITE_TOKEN` |

---

## 12. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | **yes** | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | no | `fab-search` | Pinecone index name |
| `HF_SPACE_ID` | no | `knayirp/fab-search` | HuggingFace Space ID |
| `HF_TOKEN` | no | — | HF token (only needed for private Spaces) |
| `BLOB_READ_WRITE_TOKEN` | **yes** | — | Vercel Blob read-write token |

---

## 13. Scale Estimates

| Metric | Estimate |
|---|---|
| Catalogs | ~5 current, potentially dozens |
| Pages per catalog | ~50–200 |
| Regions/tiles per page | ~5–20 (contour) or ~6–20 (tiling fallback) |
| Total vectors | ~10K–50K |
| Vector dimensions | 512 (ViT-B-32) |
| Search latency | ~1–3 s (HF Space ~0.5–2 s + Pinecone ~50 ms) |
| Indexing time per page | ~3–10 s (HF Space call + blob upload) |

---

## 14. Deployment

### 14.1 Vercel

- Runtime: Python (via `@vercel/python`)
- Entry point: `main.py` → FastAPI `app`
- Static files: `public/` served directly
- Routing defined in `vercel.json`:
  - `/api/*` → FastAPI handler
  - Static assets (`.js`, `.css`, `.png`, etc.) → `public/`
  - All other routes → FastAPI (serves `index.html`)

### 14.2 Environment Variables on Vercel

Set via Vercel Dashboard → Project → Settings → Environment Variables:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `HF_SPACE_ID`
- `BLOB_READ_WRITE_TOKEN`

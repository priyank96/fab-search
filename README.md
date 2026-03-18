# fab-search

Serverless fabric pattern search — match phone photos of fabric against PDF catalogs using CLIP visual similarity.

**Stack:** FastAPI on Vercel, CLIP embeddings via HuggingFace Space, Pinecone vector DB, Vercel Blob for image storage.

## Setup

### 1. Clone & install

```bash
cd fab-search
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=fab-search
HF_SPACE_ID=knayirp/fab-search
BLOB_READ_WRITE_TOKEN=your_vercel_blob_token
```

### 3. Run locally

```bash
uvicorn main:app --reload --port 5001
```

Open http://localhost:5001 in your browser.

## Indexing CLI

PDF indexing runs offline (not on Vercel) because it takes minutes per catalog.

Place PDFs in the `catalogs/` folder, then:

```bash
# List all indexed catalogs
python -m src.index_cli list

# Index a single PDF (re-indexes if it already exists)
python -m src.index_cli add catalogs/MyCatalog.pdf

# Remove a catalog from the index
python -m src.index_cli remove "MyCatalog.pdf"

# Full rebuild from all PDFs in catalogs/
python -m src.index_cli rebuild
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/search` | Upload an image, get matching catalog pages |
| `GET` | `/api/catalogs` | List indexed catalogs |
| `DELETE` | `/api/catalogs/{name}` | Remove a catalog |
| `GET` | `/api/health` | Health check |

### Search example (curl)

```bash
curl -X POST http://localhost:5001/api/search \
  -F "file=@photo.jpg" \
  -F "top_k=5"
```

## Project Structure

```
fab-search/
├── main.py              # FastAPI entry point
├── vercel.json          # Vercel deployment config
├── api/
│   ├── search.py        # POST /api/search
│   └── catalogs.py      # GET/DELETE /api/catalogs
├── src/
│   ├── config.py        # Constants & env vars
│   ├── embedding_client.py  # HF Space Gradio client
│   ├── pinecone_store.py    # Pinecone wrapper
│   ├── blob_store.py        # Vercel Blob uploads
│   ├── pdf_processor.py     # PDF → images (PyMuPDF)
│   ├── segment.py           # Contour-based region detection
│   ├── tiler.py             # Sliding window tiling
│   └── index_cli.py         # Indexing CLI
├── public/
│   ├── index.html       # Mobile-first frontend
│   ├── style.css
│   └── app.js
└── catalogs/            # Source PDFs (for indexing)
```

## Deployment (Vercel)

1. Push to GitHub
2. Import project in Vercel Dashboard
3. Set environment variables in Vercel → Settings → Environment Variables:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
   - `HF_SPACE_ID`
   - `BLOB_READ_WRITE_TOKEN`
4. Deploy

See [SPEC.md](SPEC.md) for full architecture details.

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.search import router as search_router
from api.catalogs import router as catalogs_router

app = FastAPI(
    title="fab-search",
    description="Serverless fabric pattern search",
    version="0.1.0",
)

# API routes
app.include_router(search_router)
app.include_router(catalogs_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve static frontend files
_public_dir = os.path.join(os.path.dirname(__file__), "public")


@app.get("/")
def index():
    return FileResponse(os.path.join(_public_dir, "index.html"))


app.mount("/", StaticFiles(directory=_public_dir), name="static")
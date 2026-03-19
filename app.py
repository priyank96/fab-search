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



@app.get("/")
def index():
    _fn_root = os.path.dirname(os.path.abspath(__file__))
    # Check common locations for static/public folders
    candidates = [
        os.path.join(_fn_root, "public"),
        os.path.join(_fn_root, "static"),
        "/var/task/public",
        "/var/task/static",
        "/var/task",
        "public/",
        "/public/"
    ]
    found = {p: os.path.isdir(p) for p in candidates}
    print(f"[fab-search] __file__: {os.path.abspath(__file__)}")
    print(f"[fab-search] root: {_fn_root}")
    for path, exists in found.items():
        label = "EXISTS" if exists else "NOT FOUND"
        print(f"[fab-search] {path}: {label}")

    return {"message": "fab-search is running", "root": _fn_root, "paths": found}


# app.mount("/", StaticFiles(directory=_public_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
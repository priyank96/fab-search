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
    print(f"[fab-search] index() root: {_fn_root}")
    for dirpath, dirnames, filenames in os.walk(_fn_root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        depth = dirpath.replace(_fn_root, '').count(os.sep)
        indent = '  ' * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        for f in filenames:
            print(f"{indent}  {f}")

    return {"message": "fab-search is running"}


# app.mount("/", StaticFiles(directory=_public_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
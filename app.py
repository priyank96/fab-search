from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.search import router as search_router
from api.catalogs import router as catalogs_router
from api.pages import router as pages_router

app = FastAPI(
    title="fab-search",
    description="Serverless fabric pattern search",
    version="0.1.0",
)

# API routes
app.include_router(search_router)
app.include_router(catalogs_router)
app.include_router(pages_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}



@app.get("/")
def index():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/index.html")


# app.mount("/", StaticFiles(directory=_public_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
api/main.py
============
FastAPI app factory. Run locally with:

    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.database import init_db
from api.routers import contracts, risk, search

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Powered Contract Intelligence & Risk Scoring",
        description="Phase 3: upload, process, semantically search, and risk-score contracts.",
        version="0.3.0",
    )

    app.include_router(contracts.router)
    app.include_router(search.router)
    app.include_router(risk.router)

    @app.on_event("startup")
    async def on_startup():
        await init_db()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def ui_index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()

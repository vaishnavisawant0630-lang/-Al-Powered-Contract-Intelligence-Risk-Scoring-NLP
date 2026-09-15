"""
api/main.py
===========
FastAPI application entry point.

Run locally with:

    uvicorn api.main:app --reload

Then open:

    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.database import init_db
from api.routers import contracts, risk, search


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Powered Contract Intelligence & Risk Scoring",
        description=(
            "Phase 3: upload, process, semantically search, "
            "and risk-score contracts."
        ),
        version="0.3.0",
    )

    # -----------------------------------------------------
    # Routers
    # -----------------------------------------------------

    app.include_router(contracts.router)
    app.include_router(search.router)
    app.include_router(risk.router)

    # -----------------------------------------------------
    # Startup
    # -----------------------------------------------------

    @app.on_event("startup")
    async def on_startup() -> None:
        await init_db()

    # -----------------------------------------------------
    # Health Check
    # -----------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # -----------------------------------------------------
    # Static Files
    # -----------------------------------------------------

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    # -----------------------------------------------------
    # Dashboard
    # -----------------------------------------------------

    @app.get("/")
    async def ui_index():
        return FileResponse(
            str(STATIC_DIR / "index.html")
        )

    return app


# ---------------------------------------------------------
# Application Instance
# ---------------------------------------------------------

app = create_app()
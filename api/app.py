"""
FastAPI Application Entry Point for DepthWizard V2
Configures CORS, API metadata, static file serving, and router inclusion.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import ARTIFACTS_DIR, ensure_directories_exist
from api.routes import router as api_router


def create_app() -> FastAPI:
    """Factory creating configured FastAPI app."""
    ensure_directories_exist()

    app = FastAPI(
        title="DepthWizard V2 API",
        description=(
            "Production-Grade Monocular Depth, GeoTIFF DEM Fusion, "
            "and 3D Terrain Reconstruction Engine API."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets (viewer, etc.)
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include API routes
    app.include_router(api_router)

    return app


app = create_app()

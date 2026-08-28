"""
FastAPI Application Entry Point for DepthWizard V2
Configures CORS, API metadata, static file serving, frontend dashboard, and router inclusion.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.config import ARTIFACTS_DIR, ensure_directories_exist
from api.routes import router as api_router


def create_app() -> FastAPI:
    """Factory creating configured FastAPI app."""
    ensure_directories_exist()

    app = FastAPI(
        title="DepthWizard V2 API & Tactical GIS Suite",
        description=(
            "Production-Grade Monocular Depth, GeoTIFF DEM Fusion, "
            "and 3D Terrain Reconstruction Engine API."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration - Allow all origins for external frontend integration
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

    # Mount tactical frontend assets
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    # Serve Tactical Frontend Dashboard at Root URL
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_frontend_root():
        frontend_index = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
        if frontend_index.exists():
            with open(frontend_index, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(
            content="""
            <html>
                <head><title>DepthWizard V2 API</title></head>
                <body style="background:#05034f;color:#e1e0ff;font-family:sans-serif;padding:40px;text-align:center;">
                    <h1>DepthWizard V2 Tactical API</h1>
                    <p><a style="color:#69d8cd;" href="/docs">Open Swagger API Docs</a> | <a style="color:#69d8cd;" href="/redoc">Open ReDoc</a></p>
                </body>
            </html>
            """
        )

    # Include API routes
    app.include_router(api_router)

    return app


app = create_app()

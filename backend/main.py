"""Main FastAPI application entry point for DepthWizard Depth & Metric Geometry Engine."""

import sys
from pathlib import Path

# Add project root to sys.path so 'backend' module imports resolve correctly when run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.router import router
from backend.depth.model_loader import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("depthwizard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan context."""
    logger.info("Initializing DepthWizard backend...")
    logger.info("Pre-loading DepthAnything V2 model...")
    try:
        model = model_manager.get_model()
        logger.info(f"Model loaded successfully on device: {model_manager._device}")
    except Exception as e:
        logger.warning(f"Lazy model initialization deferred: {str(e)}")
    yield
    logger.info("Shutting down DepthWizard backend...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production Depth & Metric Geometry Engine adhering strictly to scientific depth/elevation distinctions.",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(router)


@app.get("/", summary="Root index")
async def root():
    return {
        "engine": settings.api_title,
        "version": settings.api_version,
        "docs_url": "/docs",
        "health_url": f"{settings.api_prefix}/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

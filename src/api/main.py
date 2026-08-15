"""
FastAPI application for the Radar de Valorización Urbana.

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, _load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Radar de Valorización Urbana API",
    description=(
        "Sistema de predicción de valorización inmobiliaria para "
        "Quito y Guayaquil, Ecuador. Combina datos transaccionales, "
        "movilidad, imágenes satelitales, servicios urbanos e "
        "indicadores socioeconómicos."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    logger.info("Loading features data on startup...")
    df = _load_features()
    if not df.empty:
        logger.info("Loaded %d cells from features data.", len(df))
    else:
        logger.warning("No features data found. Some endpoints will return errors.")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Radar de Valorización Urbana API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/cells",
            "/predict?cell_id=&horizon=",
            "/explain?cell_id=",
            "/compare?cell_ids=",
            "/map?bbox=&year=",
            "/simulate",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

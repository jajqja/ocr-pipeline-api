"""FastAPI OCR Pipeline Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import get_settings
from app.core.logger import get_logger
from app.api.router import router

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="OCR Pipeline API using Surya for GPU-accelerated text detection and recognition",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"Detection batch size: {settings.BATCH_SIZE_DETECTION}")
    logger.info(f"Recognition batch size: {settings.BATCH_SIZE_RECOGNITION}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/api/docs",
        "version": "0.1.0",
    }


@app.get("/version")
async def version():
    """Get API version."""
    return {"version": "0.1.0"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info",
    )

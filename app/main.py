import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.services.detection import DetectionService
from app.services.recognition import RecognitionService
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr_pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle: Pre-loads deep learning weights
    on startup and releases hardware resources safely on shutdown.
    """
    logger.info("=== [Lifespan] Initializing FastAPI Server ===")
    try:
        logger.info("[Lifespan] Pre-loading Text Detection model weights...")
        DetectionService.get_predictor()

        logger.info("[Lifespan] Pre-loading Text Recognition model weights...")
        RecognitionService.get_recognition_predictor()

        logger.info(
            "=== [Lifespan] All AI models loaded successfully. Pipeline is ready! ==="
        )
    except Exception as e:
        logger.critical(
            f"[Lifespan] Critical error occurred while loading AI weights: {e}"
        )

    yield

    logger.info("=== [Lifespan] Initiating FastAPI Server Shutdown ===")
    logger.info("[Lifespan] Cleaning up resources and flushing AI runtime memory...")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[Lifespan] CUDA memory cache successfully flushed.")
    except ImportError:
        pass

    logger.info("=== [Lifespan] System has terminated safely ===")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Surya OCR Pipeline API",
        description="High-performance Batch OCR API supporting Text Detection, Recognition, and Layout Parsing.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response

    @app.get("/", tags=["Health Check"])
    async def root():
        return {
            "status": "healthy",
            "message": "Welcome to Surya OCR Pipeline API. System is running smoothly.",
            "timestamp": time.time(),
        }

    # Đăng ký cụm API Routers
    from app.api.v1.router import api_router

    app.include_router(api_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    logger.info("Launching Uvicorn ASGI server...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )

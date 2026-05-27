"""Health check endpoints."""

from fastapi import APIRouter, HTTPException
import torch

from app.schemas import HealthResponse
from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status and GPU info
    """
    try:
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available else "cpu"

        return HealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            device=device,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

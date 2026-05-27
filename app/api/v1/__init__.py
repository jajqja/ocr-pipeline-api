"""API v1 router - includes all v1 endpoints."""

from fastapi import APIRouter

from app.api.v1 import health, detection, recognition, parser

router = APIRouter(prefix="/api/v1")

# Include all route modules
router.include_router(health.router)
router.include_router(detection.router)
router.include_router(recognition.router)
router.include_router(parser.router)

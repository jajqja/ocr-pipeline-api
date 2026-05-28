from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.detection import router as detection_router
from app.api.v1.recognition import router as recognition_router
from app.api.v1.parser import router as parser_router

api_router = APIRouter()

# Include all route modules
api_router.include_router(health_router, tags=["health"])
api_router.include_router(detection_router, prefix="/api/v1", tags=["detection"])
api_router.include_router(recognition_router, prefix="/api/v1", tags=["recognition"])
api_router.include_router(parser_router, prefix="/api/v1", tags=["parser"])

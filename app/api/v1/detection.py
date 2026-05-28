"""Text detection API endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.detection import (
    DetectionResponse,
    DetectionRequest,
)

from app.services.detection import DetectionService
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/detection", tags=["detection"])


@router.post("", response_model=DetectionResponse)
async def detect_batch_text(request: DetectionRequest):
    """
    Detect text in batch of images with custom box padding extension.
    """
    try:
        logger.info(
            f"Text detection request received for batch size: {len(request.images_data)}"
        )

        # Router chỉ cần gọi Service và truyền dữ liệu xuống
        batch_results, processing_time = DetectionService.detect_batch(
            images_data=request.images_data,
            batch_size=request.batch_size,
            padding=request.padding,
            detector_text_threshold=request.detector_text_threshold,
            detector_blank_threshold=request.detector_blank_threshold,
        )

        return DetectionResponse(
            success=True,
            results=batch_results,
            processing_time=processing_time,
            message=f"Processed {len(batch_results)} images.",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

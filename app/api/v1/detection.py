"""Text detection API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.schemas import DetectionResponse, BBox, Polygon, TextDetection
from app.services.detection import DetectionService
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/detection", tags=["detection"])


class DetectionRequest(BaseModel):
    """Detection request model."""

    image_data: str
    batch_size: Optional[int] = None


@router.post("", response_model=DetectionResponse)
async def detect_text(request: DetectionRequest):
    """
    Detect text in image.

    Args:
        request: Detection request with base64 encoded image

    Returns:
        DetectionResponse with detected text regions
    """
    try:
        logger.info("Text detection request received")

        # Run detection
        detection_result, processing_time = DetectionService.detect(
            image_data=request.image_data,
            batch_size=request.batch_size,
        )

        # Convert detections to response format
        detections = []
        for bbox_obj in detection_result.bboxes:
            polygon = bbox_obj.polygon
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]

            detection = TextDetection(
                bbox=BBox(
                    x1=float(min(xs)),
                    y1=float(min(ys)),
                    x2=float(max(xs)),
                    y2=float(max(ys)),
                ),
                polygon=Polygon(points=polygon),
                confidence=float(bbox_obj.confidence),
            )
            detections.append(detection)

        return DetectionResponse(
            success=True,
            detections=detections,
            processing_time=processing_time,
            message=f"Detected {len(detections)} text regions",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

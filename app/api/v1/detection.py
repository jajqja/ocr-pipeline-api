"""Text detection API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.schemas import (
    DetectionResponse,
    BBox,
    Polygon,
    TextDetection,
    ImageDetectionResult,
)
from app.services.detection import DetectionService
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/detection", tags=["detection"])


class DetectionRequest(BaseModel):
    """Detection request model."""

    images_data: list[str] = Field(..., description="Base64 encoded image data")
    batch_size: Optional[int] = None


@router.post("", response_model=DetectionResponse)
async def detect_batch_text(request: DetectionRequest):
    """
    Detect text in batch of images.
    """
    try:
        logger.info(
            f"Text detection request received for batch size: {len(request.images_data)}"
        )

        # Run detection
        detection_result, processing_time = DetectionService.detect_batch(
            images_data=request.images_data,
            batch_size=request.batch_size,
        )

        batch_results = []
        total_detections_count = 0

        # Duyệt qua kết quả của từng ảnh trong batch
        for img_idx, detection in enumerate(detection_result):
            image_detections = []

            # Duyệt qua các bbox tìm thấy TRONG ẢNH NÀY
            for bbox_obj in detection.bboxes:
                polygon = bbox_obj.polygon
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]

                text_detection = TextDetection(
                    bbox=BBox(
                        x1=float(min(xs)),
                        y1=float(min(ys)),
                        x2=float(max(xs)),
                        y2=float(max(ys)),
                    ),
                    polygon=Polygon(points=polygon),
                    confidence=float(bbox_obj.confidence),
                )
                image_detections.append(text_detection)

            # Gom kết quả của ảnh này lại và append vào kết quả tổng của batch
            img_result = ImageDetectionResult(
                image_index=img_idx, detections=image_detections
            )
            batch_results.append(img_result)
            total_detections_count += len(image_detections)

        return DetectionResponse(
            success=True,
            results=batch_results,
            processing_time=processing_time,
            message=f"Processed {len(detection_result)} images. Total {total_detections_count} text regions detected.",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

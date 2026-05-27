"""Full OCR parser API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.schemas.parser import ParserResponse
from app.services.parser import ParserService
from app.core.logger import get_logger
from surya.common.surya.schema import TaskNames

logger = get_logger(__name__)

router = APIRouter(prefix="/parser", tags=["parser"])


class ParserRequest(BaseModel):
    """Parser request model."""

    image_data: str
    task_name: str = TaskNames.ocr_with_boxes
    detect_batch_size: Optional[int] = None
    recognize_batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    math_mode: bool = True
    confidence_threshold: Optional[float] = None


@router.post("", response_model=ParserResponse)
async def parse_document(request: ParserRequest):
    """
    Run full OCR pipeline: detection + recognition.

    Args:
        request: Parser request with base64 encoded image

    Returns:
        ParserResponse with detections, text lines, and full text
    """
    try:
        logger.info("Full OCR parse request received")

        # Run full pipeline
        detections, text_lines, full_text, processing_time = (
            ParserService.parse_document(
                image_data=request.image_data,
                task_name=request.task_name,
                detect_batch_size=request.detect_batch_size,
                recognize_batch_size=request.recognize_batch_size,
                max_tokens=request.max_tokens,
                math_mode=request.math_mode,
                confidence_threshold=request.confidence_threshold,
            )
        )

        return ParserResponse(
            success=True,
            detections=detections,
            text_lines=text_lines,
            full_text=full_text,
            processing_time=processing_time,
            message=f"Detected {len(detections)} regions and recognized {len(text_lines)} lines",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Parser error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

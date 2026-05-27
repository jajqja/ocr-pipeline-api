"""Text recognition API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.schemas.recognition import RecognitionResponse
from app.services.recognition import RecognitionService
from app.core.logger import get_logger
from surya.common.surya.schema import TaskNames

logger = get_logger(__name__)

router = APIRouter(prefix="/recognition", tags=["recognition"])


class RecognitionRequest(BaseModel):
    """Recognition request model."""

    image_data: str
    task_name: str = TaskNames.ocr_with_boxes
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    math_mode: bool = True


class RecognitionWithBboxesRequest(BaseModel):
    """Recognition request with bounding boxes."""

    image_data: str
    bboxes: list[list[list[int]]]
    task_name: str = TaskNames.ocr_with_boxes
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None


@router.post("", response_model=RecognitionResponse)
async def recognize_text(request: RecognitionRequest):
    """
    Recognize text in image using full pipeline.

    Args:
        request: Recognition request with base64 encoded image

    Returns:
        RecognitionResponse with recognized text
    """
    try:
        logger.info("Text recognition request received")

        # Run recognition
        text_lines, processing_time = RecognitionService.recognize_from_image(
            image_data=request.image_data,
            task_name=request.task_name,
            batch_size=request.batch_size,
            max_tokens=request.max_tokens,
            math_mode=request.math_mode,
        )

        # Combine text
        full_text = "\n".join([line.text for line in text_lines])

        return RecognitionResponse(
            success=True,
            text_lines=text_lines,
            full_text=full_text,
            processing_time=processing_time,
            message=f"Recognized {len(text_lines)} text lines",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/with-bboxes", response_model=RecognitionResponse)
async def recognize_from_bboxes(request: RecognitionWithBboxesRequest):
    """
    Recognize text from provided bounding boxes.

    Args:
        request: Recognition request with bboxes

    Returns:
        RecognitionResponse with recognized text
    """
    try:
        logger.info(f"Text recognition from {len(request.bboxes)} bboxes requested")

        # Validate bboxes
        if not request.bboxes:
            raise ValueError("No bounding boxes provided")

        # Run recognition
        text_lines, processing_time = RecognitionService.recognize_with_bboxes(
            image_data=request.image_data,
            bboxes=request.bboxes,
            task_name=request.task_name,
            batch_size=request.batch_size,
            max_tokens=request.max_tokens,
        )

        # Combine text
        full_text = "\n".join([line.text for line in text_lines])

        return RecognitionResponse(
            success=True,
            text_lines=text_lines,
            full_text=full_text,
            processing_time=processing_time,
            message=f"Recognized {len(text_lines)} text regions",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

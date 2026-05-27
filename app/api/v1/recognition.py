"""Text recognition API endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.recognition import RecognitionResponse
from app.services.recognition import RecognitionService
from app.core.logger import get_logger
from app.schemas.recognition import RecognitionRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/recognition", tags=["recognition"])


@router.post("", response_model=RecognitionResponse)
async def recognize_text(request: RecognitionRequest):
    """
    Recognize text in a batch of images using full pipeline.

    Args:
        request: Recognition request with a list of base64 encoded images

    Returns:
        RecognitionResponse with recognized text grouped by image
    """
    try:
        # Giả định request hiện tại đã đổi thành request.images_data (List[str])
        logger.info(
            f"Text recognition request received for batch size: {len(request.images_data)}"
        )

        # 1. Gọi Service xử lý Batch (đã trả về cấu trúc phân cụm theo ảnh)
        batch_results, processing_time = RecognitionService.recognize(
            images_data=request.images_data,
            bboxes=request.bboxes,
            task_name=request.task_name,
            batch_size=request.batch_size,
            max_tokens=request.max_tokens,
            math_mode=request.math_mode,
        )

        total_lines = 0
        for img_result in batch_results:
            total_lines += len(img_result.text_lines)

        return RecognitionResponse(
            success=True,
            results=batch_results,
            processing_time=processing_time,
            message=f"Processed {len(batch_results)} images. Recognized total {total_lines} text lines.",
        )

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

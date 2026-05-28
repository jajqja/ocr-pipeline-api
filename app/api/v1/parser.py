"""Full OCR parser API endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.parser import ParserRequest, DocumentParserBatchResponse
from app.services.parser import ParserService
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/parser", tags=["parser"])


@router.post("", response_model=DocumentParserBatchResponse)
async def parse_document_endpoint(request: ParserRequest):
    """
    Execute full Document Parsing Pipeline (Batch Detection + Batch Recognition).
    """
    try:
        logger.info(f"Full pipeline request received for batch size: {len(request.images_data)}")

        batch_results, processing_time = ParserService.parse_document(
            images_data=request.images_data,
            task_name=request.task_name,
            detect_batch_size=request.detect_batch_size,
            recognize_batch_size=request.recognize_batch_size,
            max_tokens=request.max_tokens,
            math_mode=request.math_mode,
            padding=request.padding,
            detector_text_threshold=request.detector_text_threshold,
            detector_blank_threshold=request.detector_blank_threshold
        )

        return DocumentParserBatchResponse(
            success=True,
            results=batch_results,
            processing_time=processing_time,
            message=f"Successfully parsed {len(batch_results)} documents.",
        )

    except ValueError as e:
        logger.error(f"Invalid input configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
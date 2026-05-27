from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.bbox import TextLine
from app.schemas.detection import TextDetection


class ParserResponse(BaseModel):
    """Parser API response - combines detection and recognition."""

    success: bool
    detections: List[TextDetection]
    text_lines: List[TextLine]
    full_text: str
    processing_time: float = Field(..., description="Processing time in seconds")
    message: Optional[str] = None

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.bbox import TextLine


class RecognitionResponse(BaseModel):
    """Recognition API response."""

    success: bool
    text_lines: List[TextLine]
    full_text: str
    processing_time: float = Field(..., description="Processing time in seconds")
    message: Optional[str] = None

from typing import List, Optional
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """Image input - can be base64 encoded or URL."""

    data: str = Field(..., description="Base64 encoded image data")
    filename: Optional[str] = Field(None, description="Original filename")


class BBox(BaseModel):
    """Bounding box representation."""

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class Polygon(BaseModel):
    """Polygon representation as list of points."""

    points: List[List[float]] = Field(..., description="List of [x, y] coordinates")


class TextChar(BaseModel):
    """Single character with position."""

    text: str
    confidence: float
    bbox: Optional[BBox] = None


class TextLine(BaseModel):
    """Line of recognized text."""

    text: str
    confidence: float
    chars: List[TextChar] = []
    bbox: Optional[BBox] = None

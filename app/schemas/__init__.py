"""Request and Response schemas for OCR Pipeline API."""

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


class TextDetection(BaseModel):
    """Text detection result."""

    bbox: BBox
    polygon: Optional[Polygon] = None
    confidence: float = Field(..., description="Detection confidence score")


class ImageDetectionResult(BaseModel):
    """Kết quả detection cho từng ảnh riêng biệt trong batch"""

    image_index: int = Field(
        ..., description="Chỉ số (index) của ảnh trong batch gửi lên"
    )
    detections: List[TextDetection] = Field(
        default_list=[], description="Danh sách các vùng chữ tìm thấy trong ảnh này"
    )


class DetectionResponse(BaseModel):
    """Detection API response mới theo dạng Batch"""

    success: bool
    # Thay đổi ở đây: Trả về List các kết quả theo từng ảnh
    results: List[ImageDetectionResult]
    processing_time: float = Field(
        ..., description="Tổng thời gian xử lý tính bằng giây"
    )
    message: Optional[str] = None


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


class RecognitionResponse(BaseModel):
    """Recognition API response."""

    success: bool
    text_lines: List[TextLine]
    full_text: str
    processing_time: float = Field(..., description="Processing time in seconds")
    message: Optional[str] = None


class ParserResponse(BaseModel):
    """Parser API response - combines detection and recognition."""

    success: bool
    detections: List[TextDetection]
    text_lines: List[TextLine]
    full_text: str
    processing_time: float = Field(..., description="Processing time in seconds")
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"
    gpu_available: bool
    device: str

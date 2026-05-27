from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.bbox import BBox, Polygon


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
    detections: List[TextDetection]


class DetectionResponse(BaseModel):
    """Detection API response mới theo dạng Batch"""

    success: bool
    # Thay đổi ở đây: Trả về List các kết quả theo từng ảnh
    results: List[ImageDetectionResult]
    processing_time: float = Field(
        ..., description="Tổng thời gian xử lý tính bằng giây"
    )
    message: Optional[str] = None


class DetectionRequest(BaseModel):
    """Detection request model."""

    images_data: List[str] = Field(..., description="Base64 encoded image data")
    batch_size: Optional[int] = None

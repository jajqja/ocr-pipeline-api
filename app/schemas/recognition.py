from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.bbox import TextLine


class ImageRecognitionResult(BaseModel):
    """Kết quả Text Recognition cho từng ảnh riêng biệt trong batch"""

    image_index: int = Field(..., description="Chỉ số (index) của ảnh trong batch")
    text_lines: List[TextLine] = Field(
        default_factory=list,
        description="Danh sách các dòng chữ nhận diện được trong ảnh này",
    )


class RecognitionResponse(BaseModel):
    """Schema phản hồi tổng cho API Recognition dạng Batch"""

    success: bool
    results: List[ImageRecognitionResult] = Field(
        ..., description="Danh sách kết quả theo từng ảnh"
    )
    processing_time: float = Field(..., description="Tổng thời gian xử lý batch (giây)")
    message: Optional[str] = None

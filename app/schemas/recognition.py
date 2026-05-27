from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.bbox import TextLine

from surya.common.surya.schema import TaskNames


class RecognitionRequest(BaseModel):
    """Recognition request model."""

    images_data: List[str]
    bboxes: Optional[List[List[List[int]]]] = (
        None  # Optional bounding boxes for each image
    )
    task_name: str = TaskNames.ocr_with_boxes
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    math_mode: bool = True


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

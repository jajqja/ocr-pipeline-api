from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.bbox import BBox, Polygon
from surya.common.surya.schema import TaskNames

class ParserRequest(BaseModel):
    """Parser request model."""
    images_data: List[str]
    task_name: str = TaskNames.ocr_with_boxes
    detect_batch_size: Optional[int] = None
    recognize_batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    math_mode: bool = True
    padding: int = Field(default=0, description="Số pixel muốn mở rộng ra các phía cho bboxes")
    detector_text_threshold: Optional[float] = None
    detector_blank_threshold: Optional[float] = None

class ParserResult(BaseModel):
    text: str
    bbox: BBox
    pollygon: Polygon

class ParserResultItem(BaseModel):
    text: str = Field(..., description="Nội dung chữ nhận diện được")
    bbox: BBox
    polygon: Polygon
    confidence: float = Field(..., description="Độ tin cậy của dòng chữ này")

class ImageParserResult(BaseModel):
    image_index: int = Field(..., description="Chỉ số vị trí của ảnh trong batch")
    full_text: str = Field(..., description="Toàn bộ text của ảnh nối bằng dấu xuống dòng")
    results: List[ParserResultItem] = Field(default_factory=list, description="Danh sách các dòng chữ chi tiết")

class DocumentParserBatchResponse(BaseModel):
    success: bool
    results: List[ImageParserResult] = Field(..., description="Danh sách kết quả theo từng ảnh")
    processing_time: float = Field(..., description="Tổng thời gian xử lý toàn bộ pipeline (giây)")
    message: Optional[str] = None
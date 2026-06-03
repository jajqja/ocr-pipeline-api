"""Parser service - combines detection and recognition."""

import logging
import time
import io
import base64
from typing import List, Tuple, Optional
from PIL import Image

import torch

from surya.common.surya.schema import TaskNames

from app.schemas.parser import ImageParserResult, ParserResultItem

from app.services.detection import DetectionService
from app.services.recognition import RecognitionService

logger = logging.getLogger(__name__)


class ParserService:
    """Full OCR pipeline service combining detection and recognition."""

    @staticmethod
    def _detect_device() -> str:
        """Detect and return the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def load_image_from_base64(image_data: str) -> Image.Image:
        """Load image from base64 encoded string."""
        try:
            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            logger.error(f"Error loading image from base64: {e}")
            raise ValueError(f"Failed to load image: {e}")

    @classmethod
    def _get_image_dimensions(cls, img_str: str) -> Tuple[int, int]:
        img_data = base64.b64decode(img_str)
        with Image.open(io.BytesIO(img_data)) as img:
            return img.size

    @classmethod
    def parse_document(
        cls,
        images_data: List[str],
        highres_images_data: Optional[List[str]] = None,
        task_name: str = TaskNames.ocr_with_boxes,
        detect_batch_size: Optional[int] = None,
        recognize_batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        math_mode: bool = True,
        padding: int = 0,
        detector_text_threshold: Optional[float] = None,
        detector_blank_threshold: Optional[float] = None,
    ) -> Tuple[List[ImageParserResult], float]:
        """
        Run full OCR pipeline (Batch Mode): detection + recognition with box padding.
        """
        try:
            start_time = time.time()

            logger.info(
                f"Step 1: Running text detection on {len(images_data)} images..."
            )
            detection_results, _ = DetectionService.detect_batch(
                images_data=images_data,
                batch_size=detect_batch_size,
                padding=padding,
                detector_text_threshold=detector_text_threshold,
                detector_blank_threshold=detector_blank_threshold,
            )

            recognition_bboxes: List[List[List[int]]] = []

            for idx, img_det in enumerate(detection_results):
                img_boxes = []

                # Mặc định tỉ lệ scale là 1.0 (không đổi)
                scale_x = 1.0
                scale_y = 1.0

                # Check nếu có highres_images_data và khớp số lượng phần tử
                if highres_images_data and idx < len(highres_images_data):
                    try:
                        # Giả định bạn có hàm helper để lấy (width, height) từ data str (base64/path)
                        # Ví dụ dùng PIL: Image.open(io.BytesIO(base64.b64decode(img_str))).size
                        orig_w, orig_h = cls._get_image_dimensions(images_data[idx])
                        high_w, high_h = cls._get_image_dimensions(
                            highres_images_data[idx]
                        )

                        scale_x = high_w / orig_w
                        scale_y = high_h / orig_h
                    except Exception as e:
                        logger.warning(
                            f"Không thể tính scale factor cho ảnh index {idx}: {e}. Giữ nguyên kích thước gốc."
                        )

                for det in img_det.detections:
                    # Nhân tọa độ với scale factor và ép kiểu về int
                    x1 = int(det.bbox.x1 * scale_x)
                    y1 = int(det.bbox.y1 * scale_y)
                    x2 = int(det.bbox.x2 * scale_x)
                    y2 = int(det.bbox.y2 * scale_y)

                    img_boxes.append([x1, y1, x2, y2])

                recognition_bboxes.append(img_boxes)

            logger.info(
                "Step 2: Running text recognition on detected bounding boxes..."
            )
            recognition_results, _ = RecognitionService.recognize(
                images_data=highres_images_data if highres_images_data else images_data,
                bboxes=recognition_bboxes,  # Truyền mảng 3D chuẩn chỉ
                task_name=task_name,
                batch_size=recognize_batch_size,
                max_tokens=max_tokens,
                math_mode=math_mode,
            )

            final_batch_results = []
            total_lines_processed = 0

            for img_idx in range(len(images_data)):
                img_det_obj = detection_results[img_idx]
                img_rec_obj = recognition_results[img_idx]

                parsed_items = []
                [d.confidence for d in img_det_obj.detections]
                [r.confidence for r in img_rec_obj.text_lines]

                for det_item, rec_item in zip(
                    img_det_obj.detections, img_rec_obj.text_lines
                ):
                    item = ParserResultItem(
                        text=rec_item.text,
                        bbox=det_item.bbox,
                        polygon=det_item.polygon,
                        confidence=rec_item.confidence,
                    )
                    parsed_items.append(item)

                full_text = "\n".join([item.text for item in parsed_items])

                image_parser_result = ImageParserResult(
                    image_index=img_idx, full_text=full_text, results=parsed_items
                )
                final_batch_results.append(image_parser_result)
                total_lines_processed += len(parsed_items)

            processing_time = time.time() - start_time
            logger.info(
                f"Full pipeline completed in {processing_time:.2f}s. "
                f"Processed {len(images_data)} images, parsed total {total_lines_processed} text items."
            )

            return final_batch_results, processing_time

        except Exception as e:
            logger.error(f"Parser error in pipeline: {e}")
            raise


if __name__ == "__main__":
    import base64
    import os

    def convert_image_to_base64(image_path: str) -> str:
        """Converts a local image file to a base64 encoded string.

        Args:
            image_path (str): Path to the local image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image path does not exist: {image_path}")

        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()

            # 3. Mã hóa sang Base64 dạng bytes, sau đó .decode('utf-8') để chuyển thành chuỗi String
            base64_encoded = base64.b64encode(binary_data).decode("utf-8")

            return base64_encoded

    base64_str = convert_image_to_base64("examples/001.png")
    service = ParserService()

    res = service.parse_document([base64_str])

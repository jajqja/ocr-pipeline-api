"""Detection service using Surya OCR."""

import logging
import time
import io
import base64
from typing import List, Tuple, Optional
from PIL import Image

import torch

from surya.detection import DetectionPredictor
from app.schemas.detection import ImageDetectionResult, TextDetection
from app.schemas.bbox import BBox, Polygon

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class DetectionService:
    """Service for text detection using Surya."""

    _predictor = None
    _device = None
    _model_path = None

    @classmethod
    def _detect_device(cls) -> str:
        """
        Detect and return the best available device (CUDA, MPS or CPU).

        Returns:
            Device string: 'cuda' if CUDA available, 'mps' if MPS available, 'cpu' otherwise.
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def get_predictor(cls) -> DetectionPredictor:
        """Get or create detection predictor instance (singleton pattern)."""
        if cls._predictor is None:
            logger.info("Initializing Detection Predictor...")
            settings = get_settings()
            if cls._device is None:
                cls._device = settings.DEVICE or cls._detect_device()

            cls._model_path = settings.DETECTION_MODEL_PATH

            cls._predictor = DetectionPredictor(
                device=cls._device, checkpoint=cls._model_path
            )
        return cls._predictor

    @staticmethod
    def load_image_from_base64(image_data: str) -> Image.Image:
        """Load image from base64 encoded string."""
        try:
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            logger.error(f"Error loading image from base64: {e}")
            raise ValueError(f"Failed to load image: {e}")

    @classmethod
    def detect_batch(
        cls,
        images_data: List[str],
        batch_size: Optional[int] = None,
        padding: int = 0,
        detector_text_threshold: Optional[float] = None,
        detector_blank_threshold: Optional[float] = None,
    ) -> Tuple[List[ImageDetectionResult], float]:
        """
        Detect text in multiple images and return structured results with optional padding.
        """
        try:
            start_time = time.time()
            app_settings = get_settings()

            # Load images
            images = [cls.load_image_from_base64(img_data) for img_data in images_data]
            logger.info(f"Loaded {len(images)} images")

            # Get predictor
            predictor = cls.get_predictor()

            if batch_size is None:
                batch_size = app_settings.BATCH_SIZE_DETECTION

            from surya.settings import settings as surya_settings

            orig_text_threshold = surya_settings.DETECTOR_TEXT_THRESHOLD
            orig_blank_threshold = surya_settings.DETECTOR_BLANK_THRESHOLD

            try:
                if detector_text_threshold is not None:
                    surya_settings.DETECTOR_TEXT_THRESHOLD = detector_text_threshold
                    logger.info(
                        f"Temporary override DETECTOR_TEXT_THRESHOLD to {detector_text_threshold}"
                    )

                if detector_blank_threshold is not None:
                    surya_settings.DETECTOR_BLANK_THRESHOLD = detector_blank_threshold
                    logger.info(
                        f"Temporary override DETECTOR_BLANK_THRESHOLD to {detector_blank_threshold}"
                    )

                logger.info(f"Running batch detection on {len(images)} images...")
                detections = predictor(images, batch_size=batch_size)

            finally:
                surya_settings.DETECTOR_TEXT_THRESHOLD = orig_text_threshold
                surya_settings.DETECTOR_BLANK_THRESHOLD = orig_blank_threshold

            batch_results = []
            total_detections_count = 0

            for img_idx, detection in enumerate(detections):
                image_detections = []
                img_w, img_h = images[img_idx].size

                for bbox_obj in detection.bboxes:
                    polygon = bbox_obj.polygon
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]

                    x1_raw = min(xs)
                    y1_raw = min(ys)
                    x2_raw = max(xs)
                    y2_raw = max(ys)

                    x1_padded = max(0, x1_raw - padding)
                    y1_padded = max(0, y1_raw - padding)
                    x2_padded = min(img_w, x2_raw + padding)
                    y2_padded = min(img_h, y2_raw + padding)

                    padded_polygon = [
                        [x1_padded, y1_padded],
                        [x2_padded, y1_padded],
                        [x2_padded, y2_padded],
                        [x1_padded, y2_padded],
                    ]

                    text_detection = TextDetection(
                        bbox=BBox(
                            x1=float(x1_padded),
                            y1=float(y1_padded),
                            x2=float(x2_padded),
                            y2=float(y2_padded),
                        ),
                        polygon=Polygon(points=padded_polygon),
                        confidence=bbox_obj.confidence,
                    )
                    image_detections.append(text_detection)

                img_result = ImageDetectionResult(
                    image_index=img_idx, detections=image_detections
                )
                batch_results.append(img_result)
                total_detections_count += len(image_detections)

            processing_time = time.time() - start_time
            logger.info(
                f"Batch detection completed in {processing_time:.2f}s. "
                f"Processed {len(images)} images. Found {total_detections_count} text regions (with padding={padding}px)."
            )

            return batch_results, processing_time

        except Exception as e:
            logger.error(f"Batch detection error: {e}")
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
    service = DetectionService()

    res = service.detect_batch([base64_str])

    print(res)

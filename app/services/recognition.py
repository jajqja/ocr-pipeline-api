"""Recognition service using Surya OCR."""

import logging
import time
import io
import base64
from typing import List, Tuple, Optional
from PIL import Image

import torch

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames

from app.core.config import get_settings
from app.schemas.bbox import TextChar, TextLine, BBox
from app.schemas.recognition import ImageRecognitionResult

logger = logging.getLogger(__name__)


class RecognitionService:
    """Service for text recognition using Surya Foundation model."""

    _foundation_predictor = None
    _recognition_predictor = None
    _device = None
    _model_path = None

    @classmethod
    def _detect_device(cls) -> str:
        """
        Detect and return the best available device.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def get_foundation_predictor(cls) -> FoundationPredictor:
        """Get or create foundation predictor instance."""
        if cls._foundation_predictor is None:
            logger.info("Initializing Foundation Predictor...")
            settings = get_settings()
            if cls._device is None:
                cls._device = settings.DEVICE or cls._detect_device()
            cls._model_path = settings.RECOGNITION_MODEL_PATH

            cls._foundation_predictor = FoundationPredictor(
                device=cls._device, checkpoint=cls._model_path
            )
        return cls._foundation_predictor

    @classmethod
    def get_recognition_predictor(cls) -> RecognitionPredictor:
        """Get or create recognition predictor instance."""
        if cls._recognition_predictor is None:
            logger.info("Initializing Recognition Predictor...")
            foundation_pred = cls.get_foundation_predictor()
            cls._recognition_predictor = RecognitionPredictor(foundation_pred)
        return cls._recognition_predictor

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
    def recognize(
        cls,
        images_data: List[str],
        bboxes: Optional[List[List[List[int]]]] = None,
        task_name: str = TaskNames.ocr_with_boxes,
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        math_mode: bool = True,
    ) -> Tuple[List[ImageRecognitionResult], float]:
        """
        Recognize text in image using full pipeline.

        Args:
            images_data: List of base64 encoded image data
            bboxes: List of bounding boxes for each image [[[x1,y1,x2,y2], ...], [...]]
            task_name: Task name (ocr_with_boxes, ocr_without_boxes, etc.)
            batch_size: Batch size for recognition
            max_tokesn: Maximum tokens for generation (if applicable)
            math_mode: Whether to enable math mode

        Returns:
            Tuple of (text lines, processing time)
        """
        try:
            start_time = time.time()
            settings = get_settings()

            # Load image
            images = [cls.load_image_from_base64(image) for image in images_data]

            if not bboxes:
                bboxes = [[[0, 0, image.size[0], image.size[1]]] for image in images]

            logger.info(f"Loaded {len(images)} images")

            # Get recognizer
            recognizer = cls.get_recognition_predictor()

            if batch_size is None:
                batch_size = settings.BATCH_SIZE_RECOGNITION

            # Run recognition
            logger.info(
                f"Running batch recognition with {task_name} on {len(images)} images..."
            )
            results = recognizer(
                images,
                bboxes=bboxes,
                task_names=[task_name] * len(images),
                recognition_batch_size=batch_size,
                max_tokens=max_tokens,
                math_mode=math_mode,
            )

            batch_results = []
            total_lines_count = 0

            for img_idx, ocr_result in enumerate(results):
                text_lines = cls._convert_results(ocr_result)
                total_lines_count += len(text_lines)

                img_result = ImageRecognitionResult(
                    image_index=img_idx, text_lines=text_lines
                )
                batch_results.append(img_result)

            processing_time = time.time() - start_time
            logger.info(
                f"Batch recognition completed in {processing_time:.2f}s. "
                f"Processed {len(images)} images, recognized total {total_lines_count} lines."
            )

            return batch_results, processing_time

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            raise

    @staticmethod
    def _convert_results(ocr_result) -> List[TextLine]:
        """Convert Surya OCRResult to TextLine objects."""
        text_lines = []

        for line in ocr_result.text_lines:
            text_chars = []

            for char in line.chars:
                text_char = TextChar(
                    text=char.text, confidence=char.confidence, bbox=None
                )
                if char.bbox_valid and char.polygon:
                    # Convert polygon to bbox
                    points = char.polygon
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]

                    text_char.bbox = BBox(
                        x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)
                    )

                text_chars.append(text_char)

            # Convert polygon to bbox for line
            line_bbox = None
            if line.polygon:
                points = line.polygon
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                line_bbox = BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

            text_line = TextLine(
                text=line.text,
                confidence=line.confidence,
                chars=text_chars,
                bbox=line_bbox,
            )
            text_lines.append(text_line)

        return text_lines


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

            base64_encoded = base64.b64encode(binary_data).decode("utf-8")

            return base64_encoded

    base64_str = convert_image_to_base64("examples/002.png")
    service = RecognitionService()

    res = service.recognize([base64_str], max_tokens=1024)

    print(res)

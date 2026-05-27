"""Parser service - combines detection and recognition."""

import logging
import time
import io
import base64
from typing import List, Tuple
from PIL import Image

import torch

from surya.common.surya.schema import TaskNames

from app.core.config import get_settings
from app.schemas.bbox import TextLine, BBox, Polygon
from app.schemas.detection import TextDetection
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
    def parse_document(
        cls,
        image_data: str,
        task_name: str = TaskNames.ocr_with_boxes,
        detect_batch_size: int = None,
        recognize_batch_size: int = None,
        max_tokens: int = None,
        math_mode: bool = True,
        confidence_threshold: float = None,
    ) -> Tuple[List[TextDetection], List[TextLine], str, float]:
        """
        Run full OCR pipeline: detection + recognition.

        Args:
            image_data: Base64 encoded image
            task_name: OCR task name
            detect_batch_size: Batch size for detection
            recognize_batch_size: Batch size for recognition
            max_tokens: Maximum tokens for generation
            math_mode: Enable math mode
            confidence_threshold: Filter detections by confidence

        Returns:
            Tuple of (detections, text_lines, full_text, processing_time)
        """
        try:
            start_time = time.time()
            settings = get_settings()

            if confidence_threshold is None:
                confidence_threshold = settings.CONFIDENCE_THRESHOLD

            if detect_batch_size is None:
                detect_batch_size = settings.BATCH_SIZE_DETECTION

            if recognize_batch_size is None:
                recognize_batch_size = settings.BATCH_SIZE_RECOGNITION

            if max_tokens is None:
                max_tokens = settings.MAX_TOKENS

            # Load image
            image = cls.load_image_from_base64(image_data)
            logger.info(f"Starting full OCR pipeline on image: {image.size}")

            # Step 1: Detection
            logger.info("Step 1: Running text detection...")
            detection_result, det_time = DetectionService.detect(
                image_data, batch_size=detect_batch_size
            )

            detections = []
            for bbox_obj in detection_result.bboxes:
                if bbox_obj.confidence >= confidence_threshold:
                    # Convert polygon to BBox
                    polygon = bbox_obj.polygon
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]

                    detection = TextDetection(
                        bbox=BBox(
                            x1=float(min(xs)),
                            y1=float(min(ys)),
                            x2=float(max(xs)),
                            y2=float(max(ys)),
                        ),
                        polygon=Polygon(points=polygon),
                        confidence=float(bbox_obj.confidence),
                    )
                    detections.append(detection)

            logger.info(f"Detected {len(detections)} text regions")

            # Step 2: Recognition
            logger.info("Step 2: Running text recognition...")
            bboxes = [
                [int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)]
                for d in detections
            ]

            if bboxes:
                text_lines, rec_time = RecognitionService.recognize_with_bboxes(
                    image_data,
                    bboxes=bboxes,
                    task_name=task_name,
                    batch_size=recognize_batch_size,
                    max_tokens=max_tokens,
                )
            else:
                # No detections, try to recognize full image
                logger.warning(
                    "No detections found. Attempting full image recognition..."
                )
                text_lines, rec_time = RecognitionService.recognize_from_image(
                    image_data,
                    task_name=task_name,
                    batch_size=recognize_batch_size,
                    max_tokens=max_tokens,
                    math_mode=math_mode,
                )

            # Combine text
            full_text = "\n".join([line.text for line in text_lines])

            processing_time = time.time() - start_time
            logger.info(f"Full pipeline completed in {processing_time:.2f}s")

            return detections, text_lines, full_text, processing_time

        except Exception as e:
            logger.error(f"Parser error: {e}")
            raise

    @classmethod
    def parse_document_batch(
        cls,
        images_data: List[str],
        task_name: str = TaskNames.ocr_with_boxes,
        detect_batch_size: int = None,
        recognize_batch_size: int = None,
        max_tokens: int = None,
        math_mode: bool = True,
    ) -> Tuple[List[Tuple[List[TextDetection], List[TextLine], str]], float]:
        """
        Run full OCR pipeline on multiple documents.

        Args:
            images_data: List of base64 encoded images
            task_name: OCR task name
            detect_batch_size: Batch size for detection
            recognize_batch_size: Batch size for recognition
            max_tokens: Maximum tokens
            math_mode: Enable math mode

        Returns:
            Tuple of (results list, total processing time)
        """
        try:
            start_time = time.time()
            results = []

            logger.info(f"Starting batch OCR pipeline on {len(images_data)} documents")

            for idx, image_data in enumerate(images_data):
                logger.info(f"Processing document {idx + 1}/{len(images_data)}")

                detections, text_lines, full_text, _ = cls.parse_document(
                    image_data,
                    task_name=task_name,
                    detect_batch_size=detect_batch_size,
                    recognize_batch_size=recognize_batch_size,
                    max_tokens=max_tokens,
                    math_mode=math_mode,
                )

                results.append((detections, text_lines, full_text))

            total_time = time.time() - start_time
            logger.info(f"Batch pipeline completed in {total_time:.2f}s")

            return results, total_time

        except Exception as e:
            logger.error(f"Batch parser error: {e}")
            raise

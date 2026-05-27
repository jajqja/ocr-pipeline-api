"""Detection service using Surya OCR."""

import logging
import time
import io
import base64
from typing import List, Tuple
from PIL import Image

import torch

from surya.detection import DetectionPredictor
from surya.detection.schema import TextDetectionResult

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
    def detect(
        cls,
        image_data: str,
        batch_size: int = None,
    ) -> Tuple[TextDetectionResult, float]:
        """
        Detect text in image.

        Args:
            image_data: Base64 encoded image data
            batch_size: Batch size for detection

        Returns:
            Tuple of (detection result, processing time)
        """
        try:
            start_time = time.time()
            settings = get_settings()

            # Load image
            image = cls.load_image_from_base64(image_data)
            logger.info(f"Image loaded: {image.size}")

            # Get predictor
            predictor = cls.get_predictor()

            # Set batch size
            if batch_size is None:
                batch_size = settings.BATCH_SIZE_DETECTION

            # Run detection
            logger.info("Running detection...")
            detections = predictor([image], batch_size=batch_size)

            processing_time = time.time() - start_time
            logger.info(
                f"Detection completed in {processing_time:.2f}s. Found {len(detections[0].bboxes)} text regions."
            )

            return detections[0], processing_time

        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise

    @classmethod
    def detect_batch(
        cls,
        images_data: List[str],
        batch_size: int = None,
    ) -> Tuple[List[TextDetectionResult], float]:
        """
        Detect text in multiple images.

        Args:
            images_data: List of base64 encoded images
            batch_size: Batch size for detection

        Returns:
            Tuple of (detection results list, processing time)
        """
        try:
            start_time = time.time()
            settings = get_settings()

            # Load images
            images = [cls.load_image_from_base64(img_data) for img_data in images_data]
            logger.info(f"Loaded {len(images)} images")

            # Get predictor
            predictor = cls.get_predictor()

            # Set batch size
            if batch_size is None:
                batch_size = settings.BATCH_SIZE_DETECTION

            # Run detection
            logger.info(f"Running batch detection on {len(images)} images...")
            detections = predictor(images, batch_size=batch_size)

            processing_time = time.time() - start_time
            total_boxes = sum(len(det.bboxes) for det in detections)
            logger.info(
                f"Batch detection completed in {processing_time:.2f}s. Found {total_boxes} text regions."
            )

            return detections, processing_time

        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            raise


# For backward compatibility
class TextDetectionService:
    """Legacy class - use DetectionService instead."""

    def __init__(self, model_path: str = None) -> None:
        self.model_path = model_path
        self.device = DetectionService._detect_device()
        self.model = DetectionService.get_predictor()
        self._settings = get_settings()
        logger.info(f"TextDetectionService initialized on device: {self.device}")

    def _detect_device(self) -> str:
        """Detect and return the best available device."""
        return DetectionService._detect_device()

    def _load_model(self):
        """Load model."""
        return DetectionService.get_predictor()

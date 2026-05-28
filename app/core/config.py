"""Configuration settings for OCR Pipeline API."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application settings
    APP_NAME: str = "OCR Pipeline API"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # Model settings - Paths to model directories
    MODEL_BASE_PATH: str = "model_path"
    DETECTION_MODEL_PATH: str = "model_path/text_detection"
    RECOGNITION_MODEL_PATH: str = "model_path/text_recognition"

    # GPU/Device settings
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    BATCH_SIZE_DETECTION: int = 4
    BATCH_SIZE_RECOGNITION: int = 8

    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    API_KEY: str | None = None

    # Allowed MIME types for image upload
    ALLOWED_MIME_TYPES: list[str] = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/gif",
        "image/bmp",
        "image/webp",
    ]

    # Model inference settings
    DETECTION_BATCH_SIZE: int = 32
    RECOGNITION_BATCH_SIZE: int = 256


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

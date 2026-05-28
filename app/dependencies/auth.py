"""API Key authentication dependency."""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """
    Verify API key from X-API-Key header.

    If API_KEY is not configured, authentication is disabled.
    If API_KEY is configured, the header must match.

    Returns:
        The API key if valid, None if auth is disabled.

    Raises:
        HTTPException: If API key is missing or invalid.
    """
    settings = get_settings()

    # If API_KEY is not set, auth is disabled
    if not settings.API_KEY:
        return None

    # API_KEY is set, so we require authentication
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key

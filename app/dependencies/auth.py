"""Authentication and security utilities."""

from typing import Optional
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthCredentials] = Depends(security),
):
    """
    Verify API key from request.

    Can be used to protect endpoints:
    @router.get("/protected", dependencies=[Depends(verify_api_key)])
    """
    # Implement your API key verification logic here
    # For now, this is a placeholder
    if credentials is None:
        # Allow requests without API key for now
        return None

    api_key = credentials.credentials
    # TODO: Verify API key against stored keys

    return api_key

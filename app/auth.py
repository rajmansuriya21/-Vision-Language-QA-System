from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from the request header.
    
    Args:
        api_key: API key from request header
    
    Returns:
        str: Verified API key
    
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing"
        )
    
    # Get API key from environment variable
    valid_api_key = os.getenv("API_KEY")
    
    if not valid_api_key:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server"
        )
    
    if api_key != valid_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key 
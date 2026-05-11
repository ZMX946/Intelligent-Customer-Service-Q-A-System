# -*- coding: utf-8 -*-
"""
API Key 鉴权模块
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from config import API_KEYS

_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(_header)) -> str:
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key. "
                   "Please provide a valid X-API-Key header."
        )
    return api_key

from __future__ import annotations

import re

from fastapi import HTTPException, Request, UploadFile

from app.core.state import AppState

SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")


def get_state(request: Request) -> AppState:
    return request.app.state.melody


async def read_upload(file: UploadFile, max_mb: float) -> bytes:
    """Read an upload in chunks, refusing anything larger than ``max_mb``."""
    limit = int(max_mb * 1024 * 1024)
    chunks, size = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"File too large (limit {max_mb:.0f} MB)")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "Empty file")
    return data


def safe_id(value: str, what: str = "id") -> str:
    if not SAFE_ID.match(value or "") or ".." in value:
        raise HTTPException(400, f"Invalid {what}")
    return value

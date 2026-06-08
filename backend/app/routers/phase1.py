import base64
import json
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException

from app.models.schemas import Phase1Request, Phase1Response
from app.services import gemini_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase1", tags=["phase1"])


def _session_dir(session_id: str) -> Path:
    from app.config import get_settings
    return Path(get_settings().tmp_dir) / session_id


def _get_image_b64_for_session(session_id: str) -> str:
    """
    Resolve the best available image for a session:
    1. selected_page.png  (PDF page the user chose)
    2. original.png       (direct image upload or normalised PNG)
    3. original.jpg / .jpeg
    """
    session_dir = _session_dir(session_id)
    for candidate in ["selected_page.png", "original.png", "original.jpg", "original.jpeg"]:
        p = session_dir / candidate
        if p.exists():
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    raise FileNotFoundError(f"No source image found for session {session_id}")


@router.post("/process", response_model=Phase1Response)
async def process_phase1(body: Phase1Request):
    """
    Run Phase 1 processing:
    - Call A: Gemini cleans the floor plan image (returns base64 PNG).
    - Call B: Gemini extracts semantic metadata (returns JSON).

    Both calls are executed in parallel via asyncio.gather.
    Results are persisted to the session directory.
    """
    session_dir = _session_dir(body.session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        image_b64 = _get_image_b64_for_session(body.session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        cleaned_b64, metadata, processing_time_ms = await gemini_service.process_floor_plan(
            image_b64
        )
    except Exception as e:
        logger.error(f"Gemini processing failed for session {body.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {e}")

    # Persist results to disk
    if cleaned_b64:
        cleaned_path = session_dir / "phase1_cleaned.png"
        async with aiofiles.open(cleaned_path, "wb") as f:
            await f.write(base64.b64decode(cleaned_b64))

    metadata_path = session_dir / "phase1_metadata.json"
    async with aiofiles.open(metadata_path, "w") as f:
        await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))

    return Phase1Response(
        cleaned_image_b64=cleaned_b64,
        metadata=metadata,
        processing_time_ms=processing_time_ms,
    )

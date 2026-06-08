"""
Phase 3 router stub — Calibration / frontend editor.

This router will be fully implemented in Phase 3.
"""
from fastapi import APIRouter
from app.models.schemas import Phase3Request, Phase3Response

router = APIRouter(prefix="/api/phase3", tags=["phase3"])


@router.post("/calibrate", response_model=Phase3Response)
async def calibrate(body: Phase3Request):
    """
    Stub endpoint for Phase 3 calibration.
    Will expose scale and geometry correction controls once implemented.
    """
    return Phase3Response()

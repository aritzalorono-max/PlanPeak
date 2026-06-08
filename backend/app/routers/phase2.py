"""
Phase 2 router stub — OpenCV/Shapely vectorization.

This router will be fully implemented in Phase 2.
"""
from fastapi import APIRouter
from app.models.schemas import Phase2Request, Phase2Response

router = APIRouter(prefix="/api/phase2", tags=["phase2"])


@router.post("/vectorize", response_model=Phase2Response)
async def vectorize(body: Phase2Request):
    """
    Stub endpoint for Phase 2 vectorization.
    Will accept a session_id and return vectorized geometry once implemented.
    """
    return Phase2Response()

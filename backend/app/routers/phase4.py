"""
Phase 4 router stub — DXF generation (ezdxf).

This router will be fully implemented in Phase 4.
"""
from fastapi import APIRouter
from app.models.schemas import Phase4Request, Phase4Response

router = APIRouter(prefix="/api/phase4", tags=["phase4"])


@router.post("/generate", response_model=Phase4Response)
async def generate_dxf(body: Phase4Request):
    """
    Stub endpoint for Phase 4 DXF generation.
    Will accept a session_id and return a downloadable DXF file once implemented.
    """
    return Phase4Response()

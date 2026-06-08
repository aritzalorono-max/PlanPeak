"""
Phase 2 stub — OpenCV/Shapely vectorization.

This module will be implemented in Phase 2. It will:
- Accept the cleaned binary PNG from Phase 1
- Use OpenCV to detect contours for walls, doors, and windows
- Use Shapely to build polygonal geometry from the detected contours
- Return a list of geometric primitives (lines, polylines, arcs) ready for DXF export
"""
import logging

logger = logging.getLogger(__name__)


def vectorize_floor_plan(cleaned_image_b64: str, metadata: dict) -> dict:
    """
    Stub: Convert a cleaned floor plan PNG to vector geometry.

    Args:
        cleaned_image_b64: Base64-encoded cleaned PNG from Phase 1.
        metadata: Semantic metadata dict from Phase 1.

    Returns:
        Dict with vectorization results (stub returns placeholder).
    """
    logger.info("Phase 2 vectorizer called — not yet implemented.")
    return {
        "status": "not_implemented",
        "message": "Phase 2 (OpenCV/Shapely vectorization) is not yet implemented.",
        "walls": [],
        "openings": [],
        "rooms": [],
    }

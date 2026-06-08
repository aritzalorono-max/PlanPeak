"""
Phase 4 stub — DXF generation using ezdxf.

This module will be implemented in Phase 4. It will:
- Accept vector geometry from Phase 2 and calibration data from Phase 3
- Use ezdxf to generate a standards-compliant DXF file
- Organise geometry into named layers (load_bearing_walls, partition_walls,
  doors, windows, rooms, annotations)
- Return the DXF file as bytes or write it to disk
"""
import logging

logger = logging.getLogger(__name__)


def generate_dxf(vector_data: dict, calibration: dict, output_path: str) -> str:
    """
    Stub: Generate a DXF file from vectorized floor plan data.

    Args:
        vector_data: Output from Phase 2 vectorizer.
        calibration: Scale and calibration data from Phase 3.
        output_path: Destination path for the generated .dxf file.

    Returns:
        Path to the generated DXF file (stub returns empty string).
    """
    logger.info("Phase 4 DXF generator called — not yet implemented.")
    return ""

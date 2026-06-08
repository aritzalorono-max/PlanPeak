"""
processing.py — minimal pre-processing for Phase 1.

Pipeline:
  1. Decode image
  2. Grayscale + CLAHE contrast enhancement (black=black, white=white)
  3. Re-encode as PNG for Gemini

No skeletonization, no HoughLinesP, no thresholding.
Gemini receives a clean high-contrast version of the original.
"""

import base64
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def prepare_for_gemini(
    image_b64: str,
    debug_dir: Optional[Path] = None,
) -> Tuple[str, Tuple[int, int]]:
    """
    Minimal pre-processing: grayscale + CLAHE contrast boost.

    Returns:
        prepared_b64  – base64 PNG ready for Gemini
        image_shape   – (height, width) of the image
    """
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    logger.info(f"[processing] image {w}×{h}")

    # CLAHE: enhances local contrast so walls are clearly darker than background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel PNG (Gemini handles color fine)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    if debug_dir is not None:
        debug_path = Path(debug_dir) / "debug_prepared.png"
        cv2.imwrite(str(debug_path), enhanced_bgr)
        logger.info(f"[processing] DEBUG prepared image saved → {debug_path}")

    ok, buf = cv2.imencode(".png", enhanced_bgr)
    prepared_b64 = base64.b64encode(buf).decode("utf-8") if ok else image_b64

    return prepared_b64, (h, w)


def validate_openings(
    metadata: dict,
    image_shape: Tuple[int, int],
    min_area_px: int = 50,
) -> dict:
    """
    Basic validation: remove openings with zero or tiny bboxes.
    """
    h, w = image_shape
    valid, n_removed = [], 0

    for op in metadata.get("openings", []):
        bbox = op.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_area_px:
            n_removed += 1
            continue
        valid.append(op)

    logger.info(f"[validate] kept={len(valid)}, removed_tiny={n_removed}")
    return {**metadata, "openings": valid}


def render_structural(
    wall_segments: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    metadata: dict,
) -> Optional[str]:
    """
    Render structural floor plan from wall segments + openings.

    - White background
    - Black lines (3px) for each wall segment
    - Red semi-transparent rectangles: doors / sliding doors
    - Blue semi-transparent rectangles: windows
    """
    try:
        import io
        from PIL import Image, ImageDraw

        h, w = image_shape
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        for x1, y1, x2, y2 in wall_segments:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=3)

        pil = Image.fromarray(canvas).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for op in metadata.get("openings", []):
            bbox = op.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            otype = op.get("type", "")
            color = (0, 80, 220, 200) if otype == "window" else (220, 30, 30, 200)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], fill=color)

        result = Image.alpha_composite(pil, overlay).convert("RGB")
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"[render] failed: {e}", exc_info=True)
        return None

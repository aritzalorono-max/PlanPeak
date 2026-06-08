"""
processing.py — image preparation for Phase 1.

Only operation: resize to max 1024px (longest side) preserving aspect ratio.
No filtering, no thresholding, no CV manipulation.
Gemini receives the original floor plan image.
"""

import base64
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_SIDE = 1024


def prepare_for_gemini(
    image_b64: str,
    debug_dir: Optional[Path] = None,
) -> Tuple[str, Tuple[int, int]]:
    """
    Resize to max 1024px longest side, preserve aspect ratio.

    Returns:
        prepared_b64  – base64 PNG
        image_shape   – (height, width) of the resized image
    """
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]
    scale = min(MAX_SIDE / max(h, w), 1.0)  # never upscale
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        logger.info(f"[processing] resized to {w}×{h}")
    else:
        logger.info(f"[processing] keeping original size {w}×{h}")

    if debug_dir is not None:
        cv2.imwrite(str(Path(debug_dir) / "debug_prepared.png"), img)

    ok, buf = cv2.imencode(".png", img)
    prepared_b64 = base64.b64encode(buf).decode("utf-8") if ok else image_b64
    return prepared_b64, (h, w)


def validate_openings(
    metadata: dict,
    image_shape: Tuple[int, int],
    min_area_px: int = 50,
) -> dict:
    """Remove openings with zero or tiny bboxes."""
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
        if (x2 - x1) * (y2 - y1) < min_area_px:
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
    White canvas + black wall lines (3px) + colored opening rectangles.
    Doors/sliding doors = red, windows = blue.
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
            color = (0, 80, 220, 200) if op.get("type") == "window" else (220, 30, 30, 200)
            draw.rectangle([int(v) for v in bbox], fill=color)

        result = Image.alpha_composite(pil, overlay).convert("RGB")
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"[render] failed: {e}", exc_info=True)
        return None

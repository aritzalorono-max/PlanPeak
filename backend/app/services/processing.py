"""
processing.py — CV pre-processing pipeline for Phase 1.

Converts a raw floor plan image into a clean binary wall skeleton
suitable for Gemini labeling. No LLM involved in this stage.

Pipeline:
  1. Grayscale + adaptive threshold  → binary (all dark lines = 255)
  2. Morphological close              → seal small gaps in wall lines
  3. Erosion (3×3, 1 iter)           → destroy thin strokes (< 3px) = furniture/dims
  4. Dilation (3×3, 2 iter)          → restore wall thickness
  5. Skeletonization (scikit-image)  → reduce walls to 1px axis
  6. HoughLinesP                     → extract long straight segments only
  7. Build final binary mask         → draw Hough segments as 2px lines on white bg

The mask is saved as debug_skeleton.png in the session directory
so Railway logs can verify what reaches Gemini.
"""

import base64
import logging
import math
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize

logger = logging.getLogger(__name__)


def build_wall_skeleton(
    image_b64: str,
    debug_dir: Optional[Path] = None,
) -> Tuple[str, List[Tuple[int, int, int, int]], np.ndarray, Tuple[int, int]]:
    """
    Run the full CV pipeline on a base64 floor plan image.

    Args:
        image_b64:  base64-encoded PNG/JPG of the floor plan.
        debug_dir:  if provided, saves debug_skeleton.png here for log inspection.

    Returns:
        skeleton_b64   – base64 PNG sent to Gemini (white bg, black 1px wall lines)
        wall_lines     – list of (x1, y1, x2, y2) Hough segments
        wall_mask      – binary uint8 ndarray (255 = wall pixel)
        image_shape    – (height, width) of original image
    """
    # ── 1. Decode ─────────────────────────────────────────────────────────────
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    logger.info(f"[processing] image {w}×{h}")

    # ── 2. Adaptive threshold ─────────────────────────────────────────────────
    # blockSize=15 handles uneven lighting from scans; C=4 keeps faint wall lines
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4,
    )

    # ── 3. Morphological close — seal small gaps in wall outlines ─────────────
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)

    # ── 4. Erosion — destroy strokes < 3px (furniture, hatch, dim lines) ─────
    erode_k = np.ones((3, 3), np.uint8)
    thick_only = cv2.erode(binary, erode_k, iterations=1)

    # ── 5. Dilation — restore wall thickness ─────────────────────────────────
    dilate_k = np.ones((3, 3), np.uint8)
    thick_only = cv2.dilate(thick_only, dilate_k, iterations=2)

    # ── 6. Skeletonize (scikit-image) — reduce to 1px axis lines ─────────────
    bool_mask = thick_only > 0
    skel_bool = skimage_skeletonize(bool_mask)
    skel = (skel_bool.astype(np.uint8)) * 255

    # ── 7. HoughLinesP — keep only long wall segments ─────────────────────────
    diagonal = math.sqrt(h * h + w * w)
    min_len = max(30, int(diagonal * 0.035))   # ~3.5% of diagonal = minimum wall
    logger.info(f"[processing] HoughLinesP min_length={min_len}px")

    lines_raw = cv2.HoughLinesP(
        skel,
        rho=1, theta=np.pi / 180,
        threshold=15,
        minLineLength=min_len,
        maxLineGap=10,
    )
    wall_lines: List[Tuple[int, int, int, int]] = []
    if lines_raw is not None:
        wall_lines = [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3]))
                      for l in lines_raw]
    logger.info(f"[processing] {len(wall_lines)} wall segments detected")

    # ── 8. Build wall_mask and clean skeleton image ───────────────────────────
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    skel_canvas = np.ones((h, w), dtype=np.uint8) * 255  # white bg

    for x1, y1, x2, y2 in wall_lines:
        cv2.line(wall_mask,   (x1, y1), (x2, y2), 255, thickness=2)
        cv2.line(skel_canvas, (x1, y1), (x2, y2), 0,   thickness=1)

    # ── 9. Save debug image ───────────────────────────────────────────────────
    if debug_dir is not None:
        debug_path = Path(debug_dir) / "debug_skeleton.png"
        cv2.imwrite(str(debug_path), skel_canvas)
        logger.info(f"[processing] DEBUG skeleton saved → {debug_path}")

    # ── 10. Encode skeleton for Gemini ────────────────────────────────────────
    ok, buf = cv2.imencode(".png", skel_canvas)
    skeleton_b64 = base64.b64encode(buf).decode("utf-8") if ok else image_b64

    return skeleton_b64, wall_lines, wall_mask, (h, w)


def validate_openings(
    metadata: dict,
    wall_mask: np.ndarray,
    min_area_px: int = 50,
) -> dict:
    """
    Architectural validation layer — removes hallucinated openings.

    Rules:
      1. Opening bbox must border wall_mask pixels on ≥ 2 opposite sides.
      2. Opening bbox area must be ≥ min_area_px (removes dot-sized noise).

    Only openings passing BOTH checks are kept.
    """
    h, w = wall_mask.shape
    valid, n_area, n_float = [], 0, 0

    for op in metadata.get("openings", []):
        bbox = op.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Rule 1: area filter
        area = (x2 - x1) * (y2 - y1)
        if area < min_area_px:
            n_area += 1
            logger.info(f"[validate] removed tiny {op.get('type')} area={area}px")
            continue

        # Rule 2: must touch wall pixels on ≥ 2 sides
        m = 14
        top    = wall_mask[max(0, y1-m):y1,     x1:x2].any()
        bottom = wall_mask[y2:min(h, y2+m),      x1:x2].any()
        left   = wall_mask[y1:y2, max(0, x1-m):x1].any()
        right  = wall_mask[y1:y2, x2:min(w, x2+m)].any()
        sides  = sum([top, bottom, left, right])

        if sides >= 2:
            valid.append(op)
        else:
            n_float += 1
            logger.info(f"[validate] removed floating {op.get('type')} bbox={bbox} sides={sides}")

    logger.info(
        f"[validate] kept={len(valid)}, removed_tiny={n_area}, removed_floating={n_float}"
    )
    return {**metadata, "openings": valid}


def render_structural(
    wall_lines: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    metadata: dict,
) -> Optional[str]:
    """
    Render the final structural floor plan from data only.
    No original image pixels are used.

    Output:
      - White background
      - Black lines (3px) for each Hough wall segment
      - Red semi-transparent rectangles: doors and sliding doors
      - Blue semi-transparent rectangles: windows
    """
    try:
        import io
        from PIL import Image, ImageDraw

        h, w = image_shape
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        for x1, y1, x2, y2 in wall_lines:
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

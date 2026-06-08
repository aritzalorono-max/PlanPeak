import asyncio
import base64
import json
import logging
import time
from typing import Optional, Tuple, Any, List

import os
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
VERTEX_PROJECT = "gen-lang-client-0434074228"
VERTEX_LOCATION = "us-central1"

METADATA_PROMPT = """You are a CAD expert analyzing a skeletonized architectural floor plan.
This image contains ONLY the skeleton of structural walls — every line is 1 pixel wide.
There are NO furniture, NO dimensions, NO text. Only wall outlines.

Return ONLY a JSON object — no markdown, no explanation.

{
  "scale_references": [
    {"value": "4.50", "unit": "m", "bbox": [x1, y1, x2, y2]}
  ],
  "rooms": [
    {"type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|garage|terrace|unknown",
     "bbox": [x1, y1, x2, y2], "label": ""}
  ],
  "openings": [
    {"type": "door|sliding_door|window", "bbox": [x1, y1, x2, y2]}
  ],
  "image_dimensions": {"width": 0, "height": 0},
  "estimated_scale": "1:50"
}

RULES for openings — read carefully:
- A door gap is a BREAK in a wall line with a curved arc nearby.
- A window is a BREAK in a wall line with two parallel short lines across it.
- ONLY detect openings at actual gaps in the wall skeleton lines.
- If a shape is NOT at a gap in a wall line, it is NOT an opening.
bbox = [left, top, right, bottom] in pixels. Return ONLY valid JSON."""


def _get_client() -> genai.Client:
    sa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../service_account.json"))
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


# ── Step 1: Skeletonize + Hough wall detection ────────────────────────────────

def _morphological_skeleton(binary: "np.ndarray") -> "np.ndarray":
    """Zhang-Suen morphological thinning — reduces lines to 1px width."""
    import cv2
    import numpy as np

    img = binary.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _detect_wall_lines(skeleton: "np.ndarray", min_length: int) -> List[Tuple]:
    """
    Run HoughLinesP on the skeleton to extract long straight line segments.
    Only lines >= min_length pixels are considered wall candidates.
    Returns list of (x1, y1, x2, y2).
    """
    import cv2
    import numpy as np

    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_length,
        maxLineGap=8,
    )
    if lines is None:
        return []
    return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in lines]


def _build_skeleton_image(
    image_b64: str,
) -> Tuple[str, List[Tuple], "np.ndarray", Tuple[int, int]]:
    """
    Full skeletonization pipeline:
      1. Binarize original (dark pixels = lines)
      2. Skeletonize (all lines → 1px)
      3. HoughLinesP with adaptive min_length (5% of image diagonal)
      4. Build wall_mask from detected lines only (discard short lines = furniture)
      5. Return (skeleton_b64_for_gemini, wall_lines, wall_mask, (h, w))
    """
    import cv2
    import numpy as np
    import math

    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive min_length: walls are at least 5% of the diagonal
    diagonal = math.sqrt(h * h + w * w)
    min_length = max(30, int(diagonal * 0.04))
    logger.info(f"Image {w}×{h}, diagonal={diagonal:.0f}, min_line_length={min_length}")

    # Binarize
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize
    skeleton = _morphological_skeleton(binary)

    # Detect long wall lines
    wall_lines = _detect_wall_lines(skeleton, min_length)
    logger.info(f"HoughLinesP detected {len(wall_lines)} wall segments")

    # Build wall_mask from Hough lines only (short noise lines excluded)
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in wall_lines:
        cv2.line(wall_mask, (x1, y1), (x2, y2), 255, thickness=3)

    # Build skeleton-only image for Gemini (white bg, black 1px walls)
    skel_canvas = np.ones((h, w), dtype=np.uint8) * 255
    for x1, y1, x2, y2 in wall_lines:
        cv2.line(skel_canvas, (x1, y1), (x2, y2), 0, thickness=1)

    success, buffer = cv2.imencode(".png", skel_canvas)
    skel_b64 = base64.b64encode(buffer).decode("utf-8") if success else image_b64

    return skel_b64, wall_lines, wall_mask, (h, w)


# ── Step 2: Gemini on skeleton image ─────────────────────────────────────────

def _call_metadata(skel_b64: str) -> Any:
    try:
        client = _get_client()
        image_bytes = base64.b64decode(skel_b64)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                METADATA_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=1,
                max_output_tokens=24576,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text.strip() if response.text else ""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as je:
            logger.warning(f"Gemini JSON parse error: {je}")
            return {"error": f"JSON parse error: {je}", "raw_response": raw[:500]}
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return {"error": str(e)}


# ── Step 3: Validate openings against wall_mask ───────────────────────────────

def _validate_openings(wall_mask: "np.ndarray", metadata: dict) -> dict:
    """
    Keep only openings whose bbox borders wall_mask pixels on ≥2 sides.
    wall_mask contains only Hough-detected long wall lines — furniture excluded.
    """
    import numpy as np

    h, w = wall_mask.shape
    valid, discarded = [], 0

    for opening in metadata.get("openings", []):
        bbox = opening.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        m = 12
        sides = sum([
            wall_mask[max(0, y1-m):y1, x1:x2].any(),
            wall_mask[y2:min(h, y2+m), x1:x2].any(),
            wall_mask[y1:y2, max(0, x1-m):x1].any(),
            wall_mask[y1:y2, x2:min(w, x2+m)].any(),
        ])
        if sides >= 2:
            valid.append(opening)
        else:
            discarded += 1
            logger.info(f"Discarded floating {opening.get('type')} bbox={bbox} sides={sides}")

    logger.info(f"Openings: kept={len(valid)}, discarded={discarded}")
    metadata = dict(metadata)
    metadata["openings"] = valid
    return metadata


# ── Step 4: Render from wall lines + validated openings ───────────────────────

def _render_from_lines(
    wall_lines: List[Tuple],
    image_shape: Tuple[int, int],
    metadata: dict,
) -> Optional[str]:
    """
    Render purely from data — no original image pixels used:
    - White canvas
    - Black lines for each Hough-detected wall segment (3px thick)
    - Red rectangles: doors and sliding doors
    - Blue rectangles: windows
    """
    try:
        import cv2
        import numpy as np
        import io
        from PIL import Image, ImageDraw

        h, w = image_shape
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Draw walls
        for x1, y1, x2, y2 in wall_lines:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=3)

        # Overlay openings
        pil = Image.fromarray(canvas).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for opening in metadata.get("openings", []):
            bbox = opening.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            otype = opening.get("type", "")
            color = (0, 80, 220, 210) if otype == "window" else (220, 30, 30, 210)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], fill=color)

        result = Image.alpha_composite(pil, overlay).convert("RGB")
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"Line render failed: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def process_floor_plan(image_b64: str) -> Tuple[Any, Optional[str], int]:
    """
    Phase 1 — Skeletal Tracing pipeline:

      1. OpenCV: binarize → skeletonize (1px lines) → HoughLinesP (long walls only)
         → skeleton image for Gemini (no furniture, no dimensions)
      2. Gemini: analyze skeleton → metadata (rooms, openings)
      3. Validate openings against Hough wall lines only
      4. Render from Hough lines + openings (no original pixels ever used)

    Returns: (metadata, structural_image_b64, processing_time_ms)
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    # Step 1 (sync): skeletonize + wall detection
    skel_b64, wall_lines, wall_mask, image_shape = await loop.run_in_executor(
        None, _build_skeleton_image, image_b64
    )
    logger.info(f"Skeleton built: {len(wall_lines)} wall segments, shape={image_shape}")

    # Step 2: Gemini on skeleton image
    metadata = await loop.run_in_executor(None, _call_metadata, skel_b64)
    logger.info(
        f"Gemini: rooms={len(metadata.get('rooms', []))}, "
        f"openings={len(metadata.get('openings', []))}, "
        f"error={metadata.get('error')}"
    )

    # Step 3: validate openings
    if "error" not in metadata:
        metadata = await loop.run_in_executor(
            None, _validate_openings, wall_mask, metadata
        )

    # Step 4: render from Hough lines + validated openings
    structural_b64 = await loop.run_in_executor(
        None, _render_from_lines, wall_lines, image_shape, metadata
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return metadata, structural_b64, elapsed_ms

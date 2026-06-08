import asyncio
import base64
import json
import logging
import time
from typing import Optional, Tuple, Any

import os
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
VERTEX_PROJECT = "gen-lang-client-0434074228"
VERTEX_LOCATION = "us-central1"

METADATA_PROMPT = """You are a CAD expert analyzing an architectural floor plan drawing.
Return ONLY a JSON object — no markdown, no explanation.

{
  "scale_references": [
    {"value": "4.50", "unit": "m", "bbox": [x1, y1, x2, y2]}
  ],
  "rooms": [
    {"type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|garage|terrace|unknown", "bbox": [x1, y1, x2, y2], "label": "text label visible in room or empty string"}
  ],
  "openings": [
    {"type": "door|sliding_door|window", "bbox": [x1, y1, x2, y2]}
  ],
  "image_dimensions": {"width": 0, "height": 0},
  "estimated_scale": "1:50"
}

STRICT RULES for opening detection:
- Only detect openings PHYSICALLY INTEGRATED INTO A WALL — they are interruptions in the wall line.
- "door": a quarter-circle swing arc with a straight line at its base, set into a gap in the wall.
- "sliding_door": a rectangular gap in the wall with parallel lines inside, no arc.
- "window": a short double or triple line embedded in the wall thickness (no arc, no swing).
- NEVER mark dimension lines, measurement arrows, leader lines, or hatching as openings.
- NEVER mark elements floating in the middle of a room as openings.
- An opening that does not interrupt a wall line is NOT an opening — discard it.

bbox = pixel coordinates [left, top, right, bottom]. Be precise.
Return ONLY valid JSON."""


def _get_client() -> genai.Client:
    sa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../service_account.json"))
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


# ── Gemini metadata extraction ────────────────────────────────────────────────

def _call_metadata(image_b64: str) -> Any:
    try:
        client = _get_client()
        image_bytes = base64.b64decode(image_b64)
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
            logger.warning(f"Gemini metadata JSON parse error: {je}")
            return {"error": f"JSON parse error: {je}", "raw_response": raw[:500]}
    except Exception as e:
        logger.error(f"Gemini metadata call failed: {e}")
        return {"error": str(e)}


# ── OpenCV: detect wall mask (thick continuous lines) ────────────────────────

def _detect_wall_mask(gray: "np.ndarray") -> "np.ndarray":
    """
    Return a binary mask of wall pixels.
    Walls are thick (≥3px) dark lines. Furniture lines are thin (1-2px).
    Strategy: erode with 3x3 kernel to destroy thin lines → only thick survive.
    Then dilate back to restore wall width.
    """
    import cv2
    import numpy as np

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Erode: thin lines (1-2px) disappear, thick walls survive
    erode_k = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, erode_k, iterations=1)

    # Dilate back to original thickness + a little extra for coverage
    dilate_k = np.ones((3, 3), np.uint8)
    wall_mask = cv2.dilate(eroded, dilate_k, iterations=2)

    return wall_mask


# ── OpenCV: filter non-wall components (furniture, annotations) ───────────────

def _filter_non_wall_components(binary: "np.ndarray", wall_mask: "np.ndarray") -> "np.ndarray":
    """
    Connected-component analysis: keep only components that overlap with
    the wall mask. Everything else (furniture, annotations, hatching, etc.)
    is isolated inside rooms and does not touch any wall → discard.
    """
    import cv2
    import numpy as np

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean_mask = np.zeros_like(binary)
    kept = 0
    removed = 0

    for i in range(1, num_labels):  # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:  # skip microscopic noise
            continue
        component = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(component, wall_mask)
        if overlap.any():
            clean_mask = cv2.bitwise_or(clean_mask, component)
            kept += 1
        else:
            removed += 1

    logger.info(f"Component filter: kept={kept}, removed={removed} (furniture/annotations)")
    return clean_mask


# ── Validate openings against wall pixels ────────────────────────────────────

def _validate_openings(original_b64: str, metadata: dict) -> dict:
    """
    Discard openings whose bbox doesn't touch dark (wall) pixels on ≥2 sides.
    """
    try:
        import io
        import numpy as np
        from PIL import Image

        image_bytes = base64.b64decode(original_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.array(img)
        h, w = arr.shape
        wall_pixels = arr < 100

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

            m = 8  # margin px
            sides = sum([
                wall_pixels[max(0, y1-m):y1, x1:x2].any(),
                wall_pixels[y2:min(h, y2+m), x1:x2].any(),
                wall_pixels[y1:y2, max(0, x1-m):x1].any(),
                wall_pixels[y1:y2, x2:min(w, x2+m)].any(),
            ])
            if sides >= 2:
                valid.append(opening)
            else:
                discarded += 1
                logger.info(f"Discarded floating {opening.get('type')} at {bbox}")

        logger.info(f"Opening validation: kept={len(valid)}, discarded={discarded}")
        metadata = dict(metadata)
        metadata["openings"] = valid
        return metadata
    except Exception as e:
        logger.error(f"Opening validation failed: {e}")
        return metadata


# ── Generate clean structural image ──────────────────────────────────────────

def _generate_clean_structural(image_b64: str, metadata: dict) -> Optional[str]:
    """
    Produce a clean structural floor plan:
    1. Binarize original image
    2. Build wall mask (thick lines only)
    3. Connected-component filter: keep only components touching walls
    4. Render white background + black walls + colored door/window overlays
    """
    try:
        import cv2
        import numpy as np
        import io
        from PIL import Image, ImageDraw

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Step 1: full binary (all dark pixels)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Step 2: wall mask (thick lines survive erosion)
        wall_mask = _detect_wall_mask(gray)

        # Step 3: filter — keep only components touching walls
        structural_mask = _filter_non_wall_components(binary, wall_mask)

        # Step 4: white canvas, paint structural_mask black
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas[structural_mask > 0] = [0, 0, 0]

        # Step 5: PIL overlay for doors (red) and windows (blue)
        pil = Image.fromarray(canvas).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for opening in metadata.get("openings", []):
            bbox = opening.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            otype = opening.get("type", "")
            color = (0, 80, 220, 200) if otype == "window" else (220, 30, 30, 200)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], fill=color)

        result = Image.alpha_composite(pil, overlay).convert("RGB")
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"Clean structural generation failed: {e}", exc_info=True)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

async def process_floor_plan(image_b64: str) -> Tuple[Any, Optional[str], int]:
    """
    Phase 1 pipeline:
      1. Gemini: extract metadata (rooms, openings, scale) from original image
      2. Validate: discard openings not touching wall pixels
      3. OpenCV: connected-component filtering removes furniture/annotations
      4. Render: clean white image with walls (black) + doors (red) + windows (blue)

    Returns: (metadata, structural_image_b64, processing_time_ms)
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    metadata = await loop.run_in_executor(None, _call_metadata, image_b64)
    logger.info(
        f"Gemini: rooms={len(metadata.get('rooms', []))}, "
        f"openings={len(metadata.get('openings', []))}, "
        f"error={metadata.get('error')}"
    )

    if "error" not in metadata:
        metadata = await loop.run_in_executor(None, _validate_openings, image_b64, metadata)

    structural_b64 = await loop.run_in_executor(
        None, _generate_clean_structural, image_b64, metadata
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return metadata, structural_b64, elapsed_ms

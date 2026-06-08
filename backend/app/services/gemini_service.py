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
- Only detect openings that are PHYSICALLY INTEGRATED INTO A WALL — they are interruptions in the wall line.
- "door": a quarter-circle swing arc with a straight line at its base, set into a gap in the wall. The bbox wraps the arc + base line tightly.
- "sliding_door": a rectangular gap in the wall with one or two parallel lines inside, no arc.
- "window": a short double or triple line embedded in the wall thickness (no arc, no swing). The bbox wraps only the wall segment containing the window lines.
- NEVER mark dimension lines, measurement arrows, leader lines, or hatching as openings.
- NEVER mark elements floating in the middle of a room or touching only a dimension line as openings.
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


# ── Layer 1: OpenCV pre-processing — remove dimension lines ──────────────────

def _preprocess_remove_dimensions(image_b64: str) -> str:
    """
    Remove dimension/annotation lines from the floor plan before sending to Gemini.
    Dimension lines are thin (1-2px) long straight lines outside or around the plan.
    Wall lines are thicker — we keep them.
    """
    try:
        import cv2
        import numpy as np

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_b64

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binary: dark pixels → 255 (lines), light → 0
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        # Detect thin horizontal lines: exactly 1px tall, at least 35px wide
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Detect thin vertical lines: exactly 1px wide, at least 35px tall
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        dim_mask = cv2.bitwise_or(h_lines, v_lines)

        # Only erase lines that are truly thin (dilate and check against thick-line mask)
        # Thick lines (walls, 3+ px): erode with 3x3 kernel — if they survive, they're thick
        thick_kernel = np.ones((3, 3), np.uint8)
        thick_lines = cv2.erode(binary, thick_kernel, iterations=1)
        # Remove from dimension mask anything that overlaps with thick lines
        dim_mask = cv2.subtract(dim_mask, thick_lines)

        # Dilate slightly to cover arrow heads and tick marks
        dilate_kernel = np.ones((2, 2), np.uint8)
        dim_mask = cv2.dilate(dim_mask, dilate_kernel, iterations=2)

        # White out dimension lines in original image
        result = img.copy()
        result[dim_mask > 0] = [255, 255, 255]

        success, buffer = cv2.imencode(".png", result)
        if not success:
            return image_b64
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        logger.error(f"Dimension line removal failed: {e}")
        return image_b64


# ── Layer 2: Gemini metadata extraction ──────────────────────────────────────

def _call_metadata(image_b64: str) -> Any:
    """Call Gemini on the pre-processed image to extract structured metadata."""
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


# ── Layer 3: Architectural validation — filter floating openings ──────────────

def _validate_openings(original_b64: str, metadata: dict) -> dict:
    """
    Remove openings whose bbox doesn't intersect with actual wall pixels.
    A valid opening must touch dark (wall) pixels on at least 2 opposite sides of its bbox.
    """
    try:
        import io
        import numpy as np
        from PIL import Image

        image_bytes = base64.b64decode(original_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.array(img)
        h, w = arr.shape

        # Wall pixels: dark (< 100)
        wall_mask = arr < 100

        valid = []
        discarded = 0
        for opening in metadata.get("openings", []):
            bbox = opening.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            margin = 6  # px to look outside bbox for wall pixels
            sides = 0

            top = wall_mask[max(0, y1 - margin):y1, x1:x2]
            bottom = wall_mask[y2:min(h, y2 + margin), x1:x2]
            left = wall_mask[y1:y2, max(0, x1 - margin):x1]
            right = wall_mask[y1:y2, x2:min(w, x2 + margin)]

            if top.any():    sides += 1
            if bottom.any(): sides += 1
            if left.any():   sides += 1
            if right.any():  sides += 1

            if sides >= 2:
                valid.append(opening)
            else:
                discarded += 1
                logger.info(f"Discarded floating {opening.get('type')} bbox={bbox} (only {sides} sides touch walls)")

        if discarded:
            logger.info(f"Validation removed {discarded} floating openings, kept {len(valid)}")

        metadata = dict(metadata)
        metadata["openings"] = valid
        return metadata

    except Exception as e:
        logger.error(f"Opening validation failed: {e}")
        return metadata


# ── Structural render (PIL overlay) ──────────────────────────────────────────

def _render_structural_pil(image_b64: str, metadata: dict) -> Optional[str]:
    """
    Render structural floor plan: original image in grayscale with
    door bboxes overlaid in red and window bboxes in blue.
    """
    try:
        import io
        from PIL import Image, ImageDraw

        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for opening in metadata.get("openings", []):
            bbox = opening.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            otype = opening.get("type", "")
            color = (0, 80, 220, 180) if otype == "window" else (220, 30, 30, 180)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], fill=color)

        result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"PIL structural render failed: {e}")
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

async def process_floor_plan(image_b64: str) -> Tuple[Any, Optional[str], int]:
    """
    Phase 1 pipeline:
      1. Gemini: extract metadata from original image
      2. Validate: discard openings not touching wall pixels
      3. PIL: render structural overlay on original image

    Returns: (metadata, structural_image_b64, processing_time_ms)
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    # Layer 1: Gemini metadata on original image
    metadata = await loop.run_in_executor(None, _call_metadata, image_b64)
    logger.info(f"Gemini metadata: rooms={len(metadata.get('rooms', []))}, openings={len(metadata.get('openings', []))}, error={metadata.get('error')}")

    # Layer 2: validate openings against original image wall pixels
    if "error" not in metadata:
        metadata = await loop.run_in_executor(None, _validate_openings, image_b64, metadata)

    # Render structural overlay on original image
    structural_b64 = await loop.run_in_executor(None, _render_structural_pil, image_b64, metadata)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return metadata, structural_b64, elapsed_ms

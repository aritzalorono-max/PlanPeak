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

# Prompt sent with the Canny pre-processed image
METADATA_PROMPT = """You are a CAD expert analyzing a pre-processed architectural floor plan.
This image has been filtered so that only continuous structural lines remain visible.
Return ONLY a JSON object — no markdown, no explanation.

{
  "scale_references": [
    {"value": "4.50", "unit": "m", "bbox": [x1, y1, x2, y2]}
  ],
  "rooms": [
    {"type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|garage|terrace|unknown",
     "bbox": [x1, y1, x2, y2], "label": "text label visible in room or empty string"}
  ],
  "openings": [
    {"type": "door|sliding_door|window", "bbox": [x1, y1, x2, y2]}
  ],
  "image_dimensions": {"width": 0, "height": 0},
  "estimated_scale": "1:50"
}

RULES:
- Only detect openings that ARE GAPS/INTERRUPTIONS in thick wall lines.
  A door is a gap in a thick wall with a swing arc. A window is a gap with a double line.
- NEVER mark thin isolated lines, hatching, text, or anything inside a room as an opening.
- A gap in a thin line is NOT an opening — only gaps in thick wall lines count.
bbox = [left, top, right, bottom] in pixels. Return ONLY valid JSON."""


def _get_client() -> genai.Client:
    sa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../service_account.json"))
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


# ── Step 1: Canny pre-processing ─────────────────────────────────────────────

def _preprocess_canny(image_b64: str) -> Tuple[str, "np.ndarray", "np.ndarray"]:
    """
    Convert floor plan to a high-contrast structural image via Canny edges.
    Returns (canny_b64_for_gemini, wall_mask, original_gray).

    wall_mask: binary mask of thick-line pixels only (walls, not furniture).
    canny_b64: edge image suitable for Gemini to identify wall gaps.
    """
    import cv2
    import numpy as np

    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Wall mask: thick lines survive 3×3 erosion, thin lines die ──────────
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    erode_k = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, erode_k, iterations=1)
    dilate_k = np.ones((3, 3), np.uint8)
    wall_mask = cv2.dilate(eroded, dilate_k, iterations=2)

    # ── Canny image for Gemini ───────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # Morphological close to reconnect broken wall edges
    close_k = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k, iterations=2)

    # Dilate wall edges so they're visibly thick (helps Gemini)
    thick_k = np.ones((2, 2), np.uint8)
    canny_thick = cv2.dilate(closed, thick_k, iterations=1)

    # White background, black edges
    canny_canvas = np.ones((h, w), dtype=np.uint8) * 255
    canny_canvas[canny_thick > 0] = 0

    success, buffer = cv2.imencode(".png", canny_canvas)
    canny_b64 = base64.b64encode(buffer).decode("utf-8") if success else image_b64

    return canny_b64, wall_mask, gray


# ── Step 2: Gemini metadata on Canny image ────────────────────────────────────

def _call_metadata(canny_b64: str) -> Any:
    try:
        client = _get_client()
        image_bytes = base64.b64decode(canny_b64)
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


# ── Step 3: Validate openings against wall mask ───────────────────────────────

def _validate_openings(wall_mask: "np.ndarray", metadata: dict) -> dict:
    """
    Discard openings whose bbox doesn't border wall-mask pixels on ≥2 sides.
    Uses the thick-line wall_mask — thin furniture lines are already excluded.
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

        m = 10
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
            logger.info(f"Discarded floating {opening.get('type')} bbox={bbox} ({sides} sides)")

    logger.info(f"Opening validation: kept={len(valid)}, discarded={discarded}")
    metadata = dict(metadata)
    metadata["openings"] = valid
    return metadata


# ── Step 4: Render purely from wall_mask + validated openings ─────────────────

def _render_from_wall_mask(wall_mask: "np.ndarray", metadata: dict) -> Optional[str]:
    """
    Render the structural floor plan from data only — no original image pixels.
    - White canvas
    - Black pixels from wall_mask (thick walls only, furniture already excluded)
    - Red rectangles: doors and sliding doors
    - Blue rectangles: windows
    """
    try:
        import numpy as np
        import io
        from PIL import Image, ImageDraw

        h, w = wall_mask.shape

        # White canvas, paint walls black
        canvas = (np.ones((h, w, 3), dtype=np.uint8) * 255)
        canvas[wall_mask > 0] = [0, 0, 0]

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
        logger.error(f"Wall-mask render failed: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def process_floor_plan(image_b64: str) -> Tuple[Any, Optional[str], int]:
    """
    Phase 1 — Structural Tracing pipeline:

      1. OpenCV Canny + Morphological Close → canny_b64 (for Gemini) + wall_mask
      2. Gemini on canny image → metadata (rooms, openings, scale)
      3. Validate openings against wall_mask (discard floating ones)
      4. Render from wall_mask only: white bg + black walls + colored openings
         (no original image pixels — furniture never appears)

    Returns: (metadata, structural_image_b64, processing_time_ms)
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    # Step 1 (sync, fast): Canny pre-processing
    canny_b64, wall_mask, _gray = await loop.run_in_executor(
        None, _preprocess_canny, image_b64
    )

    # Step 2: Gemini on canny image
    metadata = await loop.run_in_executor(None, _call_metadata, canny_b64)
    logger.info(
        f"Gemini: rooms={len(metadata.get('rooms', []))}, "
        f"openings={len(metadata.get('openings', []))}, "
        f"error={metadata.get('error')}"
    )

    # Step 3: validate openings against wall_mask
    if "error" not in metadata:
        metadata = await loop.run_in_executor(
            None, _validate_openings, wall_mask, metadata
        )

    # Step 4: render purely from wall_mask + validated openings
    structural_b64 = await loop.run_in_executor(
        None, _render_from_wall_mask, wall_mask, metadata
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return metadata, structural_b64, elapsed_ms

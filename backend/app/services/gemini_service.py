import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import os
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
VERTEX_PROJECT = "gen-lang-client-0434074228"
VERTEX_LOCATION = "us-central1"

METADATA_PROMPT = """You are a metadata extractor for architectural floor plans.
Analyze this floor plan image carefully and return a JSON object with this EXACT structure.
Do not include any explanation or markdown — return ONLY the JSON object.

{
  "scale_references": [
    {"value": "4.50", "unit": "m", "bbox": [x1, y1, x2, y2], "wall_bbox": [x1, y1, x2, y2]}
  ],
  "rooms": [
    {"type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|garage|terrace|unknown", "bbox": [x1, y1, x2, y2], "label": "original text visible in image or empty string"}
  ],
  "openings": [
    {"type": "door|sliding_door|window", "bbox": [x1, y1, x2, y2], "wall_direction": "horizontal|vertical"}
  ],
  "walls": [
    {"layer": "load_bearing_wall|partition_wall", "estimated_thickness_cm": 20, "bbox": [x1, y1, x2, y2]}
  ],
  "image_dimensions": {"width": 0, "height": 0},
  "estimated_scale": "1:100",
  "north_arrow": {"detected": false, "bbox": null}
}

Rules:
- bbox coordinates are pixel positions [left, top, right, bottom] relative to the image
- Classify walls: thickness > 25cm = load_bearing_wall, thickness < 15cm = partition_wall
- For scale_references, find dimension lines (numbers with measurement marks like arrows or ticks)
- Detect ALL rooms visible in the plan
- Return ONLY valid JSON"""


def _get_client() -> genai.Client:
    # Use Vertex AI with service account if available, else fall back to API key
    sa_path = os.path.join(os.path.dirname(__file__), "../../../../service_account.json")
    sa_path = os.path.abspath(sa_path)
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def _call_metadata(image_b64: str) -> Any:
    """Call Gemini to extract floor plan metadata as structured JSON."""
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
            logger.warning(f"Gemini metadata response was not valid JSON: {je}")
            return {"error": f"JSON parse error: {je}", "raw_response": raw[:500]}

    except Exception as e:
        logger.error(f"Gemini metadata call failed: {e}")
        return {"error": str(e)}


def _clean_image_opencv(image_b64: str) -> Optional[str]:
    """
    Phase 1 image cleaning using OpenCV:
    - Convert to grayscale
    - Apply adaptive threshold to binarize
    - Remove small noise blobs
    - Return clean binary PNG as base64
    """
    try:
        import cv2
        import numpy as np

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Could not decode image for OpenCV cleaning")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to handle uneven lighting/scanning
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=10
        )

        # Remove small noise with morphological opening
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Encode back to PNG base64
        success, buffer = cv2.imencode(".png", cleaned)
        if not success:
            return None

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        logger.error(f"OpenCV cleaning failed: {e}")
        return None


def _render_structural(cleaned_b64: Optional[str], original_b64: str, metadata: dict) -> Optional[str]:
    """
    Render structural floor plan by overlaying semantic colors onto the OpenCV-cleaned image.
    - Base: cleaned B&W image (walls already accurate from OpenCV)
    - Doors (type door|sliding_door): red filled rectangles
    - Windows: blue filled rectangles
    """
    try:
        import io
        from PIL import Image, ImageDraw

        source_b64 = cleaned_b64 if cleaned_b64 else original_b64
        image_bytes = base64.b64decode(source_b64)
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        draw = ImageDraw.Draw(base_img, "RGBA")

        openings = metadata.get("openings", [])
        for opening in openings:
            bbox = opening.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            otype = opening.get("type", "")
            # Semi-transparent fill so wall structure shows through
            color = (0, 0, 200, 160) if otype == "window" else (200, 0, 0, 160)
            draw.rectangle(bbox, fill=color)

        buf = io.BytesIO()
        base_img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"Structural render failed: {e}")
        return None


async def process_floor_plan(
    image_b64: str,
) -> Tuple[Optional[str], Any, Optional[str], int]:
    """
    Run image cleaning (OpenCV) and metadata extraction (Gemini) in parallel,
    then render a structural view from the metadata.

    Returns:
        Tuple of (cleaned_image_b64, metadata_dict, structural_image_b64, processing_time_ms).
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    cleaning_future = loop.run_in_executor(None, _clean_image_opencv, image_b64)
    metadata_future = loop.run_in_executor(None, _call_metadata, image_b64)

    cleaned_b64, metadata = await asyncio.gather(cleaning_future, metadata_future)

    structural_b64 = await loop.run_in_executor(
        None, _render_structural, cleaned_b64, image_b64, metadata
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return cleaned_b64, metadata, structural_b64, elapsed_ms

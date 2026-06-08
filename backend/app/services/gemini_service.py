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
GEMINI_IMAGE_MODEL = "gemini-2.0-flash-exp"
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

STRUCTURAL_PROMPT = (
    "This is an architectural floor plan. Generate a new clean image of this same floor plan "
    "keeping ONLY the structural elements: walls, columns, doors, sliding doors, and windows. "
    "Remove everything else: dimension lines, measurements, furniture, text labels, hatching, and annotations. "
    "Color rules: walls and columns filled solid black, windows filled solid blue, "
    "doors and sliding doors filled solid red. White background. "
    "Keep the exact same geometry, scale and proportions as the original plan."
)


def _get_client() -> genai.Client:
    sa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../service_account.json"))
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
            logger.warning(f"Gemini metadata JSON parse error: {je}")
            return {"error": f"JSON parse error: {je}", "raw_response": raw[:500]}

    except Exception as e:
        logger.error(f"Gemini metadata call failed: {e}")
        return {"error": str(e)}


def _gemini_render_structural(image_b64: str) -> Optional[str]:
    """Ask Gemini to generate a structural-only floor plan image."""
    try:
        client = _get_client()
        image_bytes = base64.b64decode(image_b64)

        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                STRUCTURAL_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            logger.warning("Gemini image generation: no candidates returned")
            return None

        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return base64.b64encode(part.inline_data.data).decode("utf-8")

        # Log what we got instead
        text_parts = [p.text for p in candidate.content.parts if p.text]
        logger.warning(f"Gemini image generation returned no image. Text parts: {text_parts[:3]}")
        finish = candidate.finish_reason
        logger.warning(f"Finish reason: {finish}")
        return None

    except Exception as e:
        logger.error(f"Gemini structural render failed: {e}", exc_info=True)
        return None


async def process_floor_plan(
    image_b64: str,
) -> Tuple[Any, Optional[str], int]:
    """
    Run metadata extraction and structural image generation in parallel via Gemini.

    Returns:
        Tuple of (metadata_dict, structural_image_b64, processing_time_ms).
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    metadata_future = loop.run_in_executor(None, _call_metadata, image_b64)
    structural_future = loop.run_in_executor(None, _gemini_render_structural, image_b64)

    metadata, structural_b64 = await asyncio.gather(metadata_future, structural_future)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return metadata, structural_b64, elapsed_ms

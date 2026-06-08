import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash-exp"

CLEANING_SYSTEM_INSTRUCTION = (
    "You are a computer vision pre-processor for architectural floor plans. "
    "Your output must be a clean binary image: walls as pure black (#000000), "
    "windows as pure blue (#0000FF), doors as pure green (#00FF00), "
    "background as pure white (#FFFFFF). "
    "Remove ALL furniture, hatching patterns, text labels, and dimensions. "
    "Classify wall thickness: walls > 25cm equivalent = load_bearing_wall layer, "
    "walls < 15cm = partition_wall layer. Mark them differently: "
    "load_bearing = black (#000000), partition = dark gray (#333333). "
    "Return ONLY the base64-encoded PNG image."
)

METADATA_SYSTEM_INSTRUCTION = (
    "You are a metadata extractor for architectural floor plans. "
    "Analyze the image and return a JSON object with this exact structure:\n"
    "{\n"
    '  "scale_references": [{"value": "4.50", "unit": "m", "bbox": [x1,y1,x2,y2], "wall_bbox": [x1,y1,x2,y2]}],\n'
    '  "rooms": [{"type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|unknown", "bbox": [x1,y1,x2,y2], "label": "original text if visible"}],\n'
    '  "openings": [{"type": "door|sliding_door|window", "bbox": [x1,y1,x2,y2], "wall_direction": "horizontal|vertical"}],\n'
    '  "image_dimensions": {"width": px, "height": px},\n'
    '  "estimated_scale": "e.g. 1:100",\n'
    '  "north_arrow": {"detected": bool, "bbox": [x1,y1,x2,y2] or null}\n'
    "}\n"
    "Return ONLY valid JSON, no markdown."
)

# Safety settings — disable blocks for architectural content
_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def _get_client() -> genai.GenerativeModel:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def _build_image_part(image_b64: str) -> dict:
    """Build a Gemini inline_data image part from a base64 PNG string."""
    return {
        "inline_data": {
            "mime_type": "image/png",
            "data": image_b64,
        }
    }


def _call_cleaning(model: genai.GenerativeModel, image_b64: str) -> Optional[str]:
    """
    Call A: Ask Gemini to return a semantically cleaned floor plan image.
    Returns base64 PNG string or None on failure.
    """
    try:
        response = model.generate_content(
            [
                CLEANING_SYSTEM_INSTRUCTION,
                _build_image_part(image_b64),
                "Please clean this floor plan according to the instructions and return the base64-encoded PNG.",
            ],
            safety_settings=_SAFETY_SETTINGS,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
            ),
        )

        raw = response.text.strip() if response.text else ""

        # Strip common markdown code fences if present
        for prefix in ("```png", "```image", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        # Validate that the result looks like base64
        if raw and len(raw) > 100:
            # Attempt to decode to verify
            base64.b64decode(raw + "==", validate=False)
            return raw

        logger.warning("Gemini cleaning call returned empty or too-short data.")
        return None

    except Exception as e:
        logger.error(f"Gemini cleaning call failed: {e}")
        return None


def _call_metadata(model: genai.GenerativeModel, image_b64: str) -> Any:
    """
    Call B: Ask Gemini to extract floor plan metadata as JSON.
    Returns parsed dict or dict with 'error' key on failure.
    """
    try:
        response = model.generate_content(
            [
                METADATA_SYSTEM_INSTRUCTION,
                _build_image_part(image_b64),
                "Extract all metadata from this floor plan and return as JSON.",
            ],
            safety_settings=_SAFETY_SETTINGS,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
            ),
        )

        raw = response.text.strip() if response.text else ""

        # Strip markdown code fences if present
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as je:
            logger.warning(f"Gemini metadata response was not valid JSON: {je}")
            return {"error": f"JSON parse error: {je}", "raw_response": raw}

    except Exception as e:
        logger.error(f"Gemini metadata call failed: {e}")
        return {"error": str(e)}


async def process_floor_plan(
    image_b64: str,
) -> Tuple[Optional[str], Any, int]:
    """
    Run Gemini Call A (cleaning) and Call B (metadata) in parallel.

    Args:
        image_b64: Base64-encoded PNG of the floor plan.

    Returns:
        Tuple of (cleaned_image_b64 or None, metadata_dict, processing_time_ms).
    """
    model = _get_client()
    start = time.monotonic()

    loop = asyncio.get_event_loop()

    # Run both blocking Gemini calls concurrently in the thread pool
    cleaning_future = loop.run_in_executor(None, _call_cleaning, model, image_b64)
    metadata_future = loop.run_in_executor(None, _call_metadata, model, image_b64)

    cleaned_b64, metadata = await asyncio.gather(cleaning_future, metadata_future)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return cleaned_b64, metadata, elapsed_ms

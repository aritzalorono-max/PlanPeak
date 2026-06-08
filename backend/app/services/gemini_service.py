"""
gemini_service.py — Phase 1: furniture removal via Vertex AI image generation.

Sends the original floor plan to Gemini 2.0 Flash (Vertex AI) and asks it
to return a new image with only structural elements (walls, doors, windows).
Image generation is only available via Vertex AI, not the standard API key.
"""

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Any

from google import genai
from google.genai import types

from app.config import get_settings
from app.services.processing import get_image_dimensions

logger = logging.getLogger(__name__)

IMAGE_GEN_MODEL = "gemini-2.0-flash-preview-image-generation"
LABELING_MODEL = "gemini-2.5-flash"

CLEANING_PROMPT = (
    "Genera una nueva imagen de esta planta arquitectónica eliminando todo el mobiliario "
    "(camas, sofás, mesas, sillas, electrodomésticos, bañeras, inodoros, etc.) y las cotas "
    "de medición. Conserva únicamente los muros estructurales, columnas, ventanas y puertas. "
    "Pinta los muros y columnas en negro sobre fondo blanco. "
    "Las ventanas en azul y las puertas (incluidas correderas) en rojo. "
    "El resultado debe ser una imagen limpia de planta arquitectónica solo con la estructura."
)

LABELING_PROMPT = """Analyze this architectural floor plan image.
Return ONLY a JSON object. No markdown. No explanation.

{
  "scale_references": [{"value": "4.50", "unit": "m", "bbox": [x1, y1, x2, y2]}],
  "rooms": [{"type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|garage|terrace|unknown", "bbox": [x1, y1, x2, y2], "label": ""}],
  "openings": [{"type": "door|sliding_door|window", "bbox": [x1, y1, x2, y2]}],
  "image_dimensions": {"width": 0, "height": 0},
  "estimated_scale": "1:50"
}

bbox = [left, top, right, bottom] pixel coordinates. Return ONLY valid JSON."""


def _get_vertex_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(
        vertexai=True,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )


def _get_api_client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key)


def _remove_furniture(image_b64: str) -> Optional[str]:
    """
    Send original floor plan to Gemini via Vertex AI.
    Returns base64 PNG of the cleaned structural image, or None on failure.
    """
    try:
        client = _get_vertex_client()
        image_bytes = base64.b64decode(image_b64)

        response = client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                CLEANING_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                logger.info("[gemini] received generated image")
                return base64.b64encode(part.inline_data.data).decode("utf-8")

        logger.warning("[gemini] no image part in response")
        return None

    except Exception as e:
        logger.error(f"[gemini] image generation failed: {e}")
        return None


def _label_floor_plan(image_b64: str) -> Any:
    """
    Run semantic labeling on the (cleaned) image to extract rooms/openings/scale.
    Falls back to API key if Vertex AI is not configured.
    """
    try:
        client = _get_api_client()
        image_bytes = base64.b64decode(image_b64)
        response = client.models.generate_content(
            model=LABELING_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                LABELING_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=1,
                max_output_tokens=16384,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text.strip() if response.text else ""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as je:
            logger.warning(f"[gemini] JSON parse error: {je}")
            return {"error": str(je), "raw_response": raw[:500]}
    except Exception as e:
        logger.error(f"[gemini] labeling failed: {e}")
        return {"error": str(e)}


async def process_floor_plan(
    image_b64: str,
    session_dir: Optional[Path] = None,
) -> Tuple[Any, Optional[str], str, int]:
    """
    Phase 1 pipeline:
      1. Send original image to Vertex AI → receive cleaned image (no furniture)
      2. Run labeling on cleaned image → rooms, openings, scale
      3. Return (metadata, cleaned_image_b64, original_b64, elapsed_ms)

    If Vertex AI image generation fails, structural_b64 will be None
    and the original image is shown instead.
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    h, w = await loop.run_in_executor(None, get_image_dimensions, image_b64)
    image_shape = (h, w)
    logger.info(f"[phase1] image {w}×{h}")

    # Step 1: remove furniture via Vertex AI image generation
    logger.info("[phase1] calling Vertex AI for furniture removal...")
    structural_b64 = await loop.run_in_executor(None, _remove_furniture, image_b64)

    if structural_b64:
        logger.info("[phase1] furniture removal successful")
        label_target = structural_b64
    else:
        logger.warning("[phase1] furniture removal failed — labeling original image")
        label_target = image_b64

    # Step 2: label the cleaned (or original) image
    logger.info("[phase1] labeling floor plan...")
    metadata = await loop.run_in_executor(None, _label_floor_plan, label_target)
    logger.info(
        f"[phase1] rooms={len(metadata.get('rooms', []))}, "
        f"openings={len(metadata.get('openings', []))}, "
        f"error={metadata.get('error')}"
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(f"[phase1] done in {elapsed_ms}ms")

    # skeleton_image_b64 slot now holds the cleaned structural image
    return metadata, structural_b64, structural_b64 or image_b64, elapsed_ms

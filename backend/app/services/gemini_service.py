"""
gemini_service.py — Gemini Vision wrapper for Phase 1.

Gemini receives a contrast-enhanced version of the original floor plan
and is asked to identify structural walls (as line segments) plus
rooms and openings. No pre-filtering of the image — the model decides.
"""

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
from app.services.processing import (
    prepare_for_gemini,
    validate_openings,
    render_structural,
)

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
VERTEX_PROJECT = "gen-lang-client-0434074228"
VERTEX_LOCATION = "us-central1"

ARCHITECT_PROMPT = """Analiza esta imagen de planta arquitectónica.

Eres un arquitecto veterano con 20 años de experiencia leyendo planos de construcción.

TAREA PRINCIPAL: Identifica el layout estructural del edificio.

LOS MUROS son las líneas más GRUESAS y SÓLIDAS que definen el perímetro de las estancias y separan los espacios. Tienen un grosor visible (no son líneas finas). Forman una red cerrada que delimita habitaciones.

IGNORA COMPLETAMENTE: mobiliario (camas, sofás, mesas, sillas, electrodomésticos, bañeras, inodoros), cotas de medición, flechas, textos, líneas de dimensión (líneas finas con números), tramas de relleno decorativas.

Para cada muro estructural, devuelve las coordenadas de píxel de inicio y fin como segmento de línea [x1, y1, x2, y2].

Devuelve ÚNICAMENTE un objeto JSON válido. Sin markdown. Sin explicación.

{
  "scale_references": [
    {"value": "4.50", "unit": "m", "bbox": [x1, y1, x2, y2]}
  ],
  "rooms": [
    {
      "type": "kitchen|bathroom|bedroom|living_room|hallway|dining_room|garage|terrace|unknown",
      "bbox": [x1, y1, x2, y2],
      "label": ""
    }
  ],
  "wall_segments": [
    {"start": [x1, y1], "end": [x2, y2]}
  ],
  "openings": [
    {"type": "door|sliding_door|window", "bbox": [x1, y1, x2, y2]}
  ],
  "image_dimensions": {"width": 0, "height": 0},
  "estimated_scale": "1:50"
}

bbox = [izquierda, arriba, derecha, abajo] en coordenadas de píxel.
wall_segments: coordenadas de la línea central de cada muro.
Devuelve ÚNICAMENTE JSON válido."""


def _get_client() -> genai.Client:
    sa_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../service_account.json")
    )
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    return genai.Client(api_key=get_settings().gemini_api_key)


def _call_gemini(image_b64: str) -> Any:
    """Send image to Gemini for architect-vision analysis."""
    try:
        client = _get_client()
        image_bytes = base64.b64decode(image_b64)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ARCHITECT_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=1,
                max_output_tokens=32768,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text.strip() if response.text else ""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as je:
            logger.warning(f"[gemini] JSON parse error: {je}")
            return {"error": f"JSON parse error: {je}", "raw_response": raw[:500]}
    except Exception as e:
        logger.error(f"[gemini] call failed: {e}")
        return {"error": str(e)}


async def process_floor_plan(
    image_b64: str,
    session_dir: Optional[Path] = None,
) -> Tuple[Any, Optional[str], str, int]:
    """
    Phase 1 pipeline:
      1. CLAHE contrast enhancement (preserve image structure)
      2. Gemini architect-vision analysis (walls + rooms + openings)
      3. validate_openings (remove tiny/invalid bboxes)
      4. render_structural (white canvas + wall segments + colored openings)

    Returns: (metadata, structural_image_b64, prepared_b64, processing_time_ms)
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    # Step 1: minimal pre-processing
    logger.info("[phase1] preparing image...")
    prepared_b64, image_shape = await loop.run_in_executor(
        None,
        lambda: prepare_for_gemini(image_b64, debug_dir=session_dir),
    )
    logger.info(f"[phase1] image ready: shape={image_shape}")

    # Step 2: Gemini architect vision
    logger.info("[phase1] calling Gemini (architect vision)...")
    metadata = await loop.run_in_executor(None, _call_gemini, prepared_b64)
    logger.info(
        f"[phase1] Gemini result: rooms={len(metadata.get('rooms', []))}, "
        f"walls={len(metadata.get('wall_segments', []))}, "
        f"openings={len(metadata.get('openings', []))}, "
        f"error={metadata.get('error')}"
    )

    # Step 3: validate openings
    if "error" not in metadata:
        metadata = await loop.run_in_executor(
            None, validate_openings, metadata, image_shape
        )

    # Step 4: render structural image from wall_segments + openings
    wall_segments = [
        (int(s["start"][0]), int(s["start"][1]), int(s["end"][0]), int(s["end"][1]))
        for s in metadata.get("wall_segments", [])
        if s.get("start") and s.get("end")
    ]
    structural_b64 = await loop.run_in_executor(
        None, render_structural, wall_segments, image_shape, metadata
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(f"[phase1] done in {elapsed_ms}ms")
    return metadata, structural_b64, prepared_b64, elapsed_ms

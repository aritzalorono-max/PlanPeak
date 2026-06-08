"""
processing.py — image pass-through for Phase 1.

No filtering. The original image is sent directly to Gemini.
"""

import base64
import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_image_dimensions(image_b64: str) -> Tuple[int, int]:
    """Decode image and return (height, width)."""
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]
    logger.info(f"[processing] image {w}×{h}")
    return h, w

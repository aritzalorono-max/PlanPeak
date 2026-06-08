from pydantic import BaseModel
from typing import Optional, List, Any


# Upload schemas
class UploadResponse(BaseModel):
    session_id: str
    file_type: str  # "pdf" | "image"
    page_count: int
    preview_page_b64: str


class SelectPageRequest(BaseModel):
    session_id: str
    page_number: int


class SelectPageResponse(BaseModel):
    session_id: str
    page_image_b64: str


# Phase 1 schemas
class Phase1Request(BaseModel):
    session_id: str


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ScaleReference(BaseModel):
    value: str
    unit: str
    bbox: List[float]
    wall_bbox: Optional[List[float]] = None


class Room(BaseModel):
    type: str
    bbox: List[float]
    label: Optional[str] = None


class Opening(BaseModel):
    type: str
    bbox: List[float]
    wall_direction: str


class NorthArrow(BaseModel):
    detected: bool
    bbox: Optional[List[float]] = None


class ImageDimensions(BaseModel):
    width: int
    height: int


class FloorPlanMetadata(BaseModel):
    scale_references: List[ScaleReference] = []
    rooms: List[Room] = []
    openings: List[Opening] = []
    image_dimensions: Optional[ImageDimensions] = None
    estimated_scale: Optional[str] = None
    north_arrow: Optional[NorthArrow] = None
    error: Optional[str] = None


class Phase1Response(BaseModel):
    structural_image_b64: Optional[str]
    metadata: Any
    processing_time_ms: int


# Phase 2 stub schemas
class Phase2Request(BaseModel):
    session_id: str


class Phase2Response(BaseModel):
    status: str = "not_implemented"
    message: str = "Phase 2 (OpenCV vectorization) is not yet implemented."


# Phase 3 stub schemas
class Phase3Request(BaseModel):
    session_id: str


class Phase3Response(BaseModel):
    status: str = "not_implemented"
    message: str = "Phase 3 (Calibration editor) is not yet implemented."


# Phase 4 stub schemas
class Phase4Request(BaseModel):
    session_id: str


class Phase4Response(BaseModel):
    status: str = "not_implemented"
    message: str = "Phase 4 (DXF generation) is not yet implemented."

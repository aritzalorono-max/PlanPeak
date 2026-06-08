import logging
import os
import shutil
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import SelectPageRequest, SelectPageResponse, UploadResponse
from app.services import pdf_converter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def _session_dir(session_id: str) -> Path:
    settings = get_settings()
    return Path(settings.tmp_dir) / session_id


def _ensure_session_dir(session_id: str) -> Path:
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a PNG, JPG, or PDF file upload.

    - PDF: returns page count + low-res preview of page 1.
    - Image: saves directly, returns page_count=1 + preview of the image.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    session_id = str(uuid.uuid4())
    session_dir = _ensure_session_dir(session_id)

    original_path = session_dir / f"original{suffix}"

    # Stream file to disk
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.",
        )

    async with aiofiles.open(original_path, "wb") as f:
        await f.write(content)

    try:
        if suffix == ".pdf":
            page_count = pdf_converter.get_page_count(str(original_path))
            preview_b64 = pdf_converter.get_page_preview(str(original_path), page=0, dpi=72)
            file_type = "pdf"
        else:
            page_count = 1
            # For images, generate a small preview by re-encoding via Pillow
            from PIL import Image
            import io, base64

            img = Image.open(io.BytesIO(content))
            img.thumbnail((800, 800))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Also save a normalised PNG for consistent downstream processing
            normalised_path = session_dir / "original.png"
            if suffix != ".png":
                img_full = Image.open(io.BytesIO(content))
                img_full.save(str(normalised_path), format="PNG")
            file_type = "image"

    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        shutil.rmtree(str(session_dir), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

    return UploadResponse(
        session_id=session_id,
        file_type=file_type,
        page_count=page_count,
        preview_page_b64=preview_b64,
    )


@router.post("/select-page", response_model=SelectPageResponse)
async def select_page(body: SelectPageRequest):
    """
    Convert a specific PDF page to a high-res PNG (300 DPI) for AI processing.
    Saves result as /tmp/planpeak/{session_id}/selected_page.png.
    """
    session_dir = _session_dir(body.session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    pdf_path = session_dir / "original.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail="No PDF found for this session.")

    if body.page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1.")

    try:
        page_b64 = pdf_converter.convert_page_to_image(
            str(pdf_path), page_number=body.page_number, dpi=300
        )
    except Exception as e:
        logger.error(f"Page conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Page conversion failed: {e}")

    # Persist selected page PNG for Phase 1 to use
    import base64
    selected_path = session_dir / "selected_page.png"
    async with aiofiles.open(selected_path, "wb") as f:
        await f.write(base64.b64decode(page_b64))

    return SelectPageResponse(session_id=body.session_id, page_image_b64=page_b64)

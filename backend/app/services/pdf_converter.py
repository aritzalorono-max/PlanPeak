import base64
import io
import logging
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

logger = logging.getLogger(__name__)


def get_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF file."""
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        return info.get("Pages", 0)
    except (PDFInfoNotInstalledError, PDFPageCountError) as e:
        logger.error(f"Failed to get page count for {pdf_path}: {e}")
        raise


def _pil_to_b64(image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_page_preview(pdf_path: str, page: int = 0, dpi: int = 72) -> str:
    """
    Convert a single PDF page to a low-res base64 PNG for preview purposes.

    Args:
        pdf_path: Path to the PDF file.
        page: Zero-based page index.
        dpi: Resolution (default 72 for previews).

    Returns:
        Base64-encoded PNG string.
    """
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page + 1,
            last_page=page + 1,
        )
        if not pages:
            raise ValueError(f"No pages returned for page index {page}")
        return _pil_to_b64(pages[0])
    except Exception as e:
        logger.error(f"Failed to generate preview for page {page} of {pdf_path}: {e}")
        raise


def convert_page_to_image(pdf_path: str, page_number: int, dpi: int = 300) -> str:
    """
    Convert a single PDF page to a high-res base64 PNG for AI processing.

    Args:
        pdf_path: Path to the PDF file.
        page_number: One-based page number (as returned to the frontend).
        dpi: Resolution (default 300 for processing quality).

    Returns:
        Base64-encoded PNG string.
    """
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
        if not pages:
            raise ValueError(f"No pages returned for page number {page_number}")
        return _pil_to_b64(pages[0])
    except Exception as e:
        logger.error(
            f"Failed to convert page {page_number} of {pdf_path} to image: {e}"
        )
        raise

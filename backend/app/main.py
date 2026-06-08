import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload, phase1, phase2, phase3
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Ensure temp upload directory exists at startup
Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PlanPeak API",
    description="Convert 2D architectural floor plans (PNG/PDF) to DXF using AI + OpenCV.",
    version="0.1.0",
)

# CORS — allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(phase1.router)
app.include_router(phase2.router)
app.include_router(phase3.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

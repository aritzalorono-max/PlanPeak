# PlanPeak

Convert 2D architectural floor plans (PNG/PDF) into DXF files using AI + OpenCV.

## Architecture

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Gemini Vision — semantic cleaning + metadata extraction | **Implemented** |
| 2 | OpenCV/Shapely vectorization | Stub |
| 3 | React + Konva.js calibration editor | Stub |
| 4 | DXF generation (ezdxf) | Stub |
| 5 | 3D visualization (Three.js) | Stub |

## Quick start

### Prerequisites

- Docker & Docker Compose
- A [Google Gemini API key](https://makersuite.google.com/app/apikey)

### Setup

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set GEMINI_API_KEY=your_key_here
```

### Run with Docker

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Run locally (without Docker)

**Backend:**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Ensure poppler-utils is installed: sudo apt-get install poppler-utils
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

## Phase 1 — How it works

1. User uploads a PNG/JPG/PDF floor plan.
2. For PDFs: select a page (previewed as thumbnails).
3. Click **Process with AI** — two parallel Gemini 2.0 Flash calls run:
   - **Call A**: Returns a semantically cleaned PNG (walls, doors, windows only — no furniture/text).
   - **Call B**: Returns structured JSON metadata (rooms, openings, scale references, dimensions).
4. Results are displayed side-by-side with a metadata summary.

## Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (required) |

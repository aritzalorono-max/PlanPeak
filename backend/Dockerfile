FROM python:3.11-slim
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends poppler-utils libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

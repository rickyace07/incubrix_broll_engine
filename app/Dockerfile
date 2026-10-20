# Use an official lightweight Python 3.11 runtime
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies, FFmpeg, and fonts for subtitle rendering
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fontconfig \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-core \
    fonts-indic \
    fonts-taml \
    fonts-deva \
    curl && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition first for layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . /app/

# Ensure workspace directories exist
RUN mkdir -p /app/local_s3 /app/tmp_work /app/outputs

EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
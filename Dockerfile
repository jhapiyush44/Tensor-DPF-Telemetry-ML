# -------------------------------------------------
# Base image
# -------------------------------------------------
# Slim python image keeps size small
FROM python:3.11-slim


# -------------------------------------------------
# Environment settings
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# -------------------------------------------------
# System deps (needed for pandas/pyarrow sometimes)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*


# -------------------------------------------------
# Install Python deps first (layer caching)
# -------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# -------------------------------------------------
# Copy project files
# -------------------------------------------------
COPY . .


# -------------------------------------------------
# Expose FastAPI port
# -------------------------------------------------
EXPOSE 8000


# -------------------------------------------------
# Default command
# -------------------------------------------------
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

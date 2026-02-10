# =================================================
# Base image
# =================================================
FROM python:3.11-slim

# =================================================
# Environment
# =================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


# =================================================
# System deps (temporary for building wheels)
# =================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# =================================================
# Install Python deps (cached layer)
# =================================================
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# =================================================
# Remove build deps (shrink image ⭐)
# =================================================
RUN apt-get purge -y gcc build-essential && \
    apt-get autoremove -y


# =================================================
# Copy project
# =================================================
COPY . .


# =================================================
# Non-root user (production best practice ⭐)
# =================================================
RUN useradd -m appuser
USER appuser


# =================================================
# Expose port
# =================================================
EXPOSE 8000


# =================================================
# Start server
# =================================================
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

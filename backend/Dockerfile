FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/var/tmp \
    PYTHONPATH=/app \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PYTHONWARNINGS="ignore:pkg_resources is deprecated"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
      /var/tmp \
      /root/.cache/pip \
      /app/.cache/torch \
      /app/.cache/huggingface \
      /app/models \
 && chmod 1777 /var/tmp

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN TMPDIR=/var/tmp pip install --no-cache-dir -r /app/requirements.txt

COPY scripts /app/scripts

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

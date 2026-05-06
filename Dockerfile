# NVIDIA CUDA 12.x with cuDNN; matches RTX 5080 driver stack.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Editable install needs src/ + README.md present at install time.
# Copy these first so the pip-install layer is cacheable when only deps change.
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -e ".[dev]"

# Configs/scripts/tests change more often. They are also volume-mounted in
# docker-compose.yml for dev iteration, so this COPY is mainly for `docker run`.
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

CMD ["python", "-m", "budgetfair.runner", "--help"]

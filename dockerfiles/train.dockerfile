# Base image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

COPY src/mlopsg24/model.py src/mlopsg24/model.py
COPY src/mlopsg24/train.py src/mlopsg24/train.py
COPY src/mlopsg24/__init__.py src/mlopsg24/__init__.py
COPY data/processed data/processed/
COPY README.md README.md

ENTRYPOINT ["uv", "run", "src/mlopsg24/train.py", "--epochs", "15"]

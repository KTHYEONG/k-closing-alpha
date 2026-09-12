FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul \
    UV_LINK_MODE=copy

# lightgbm 은 OpenMP 런타임(libgomp)을 요구한다 (VPS 실측: 누락 시 ImportError).
# ripgrep 은 test_kis_client_facade_removed.py 등 코드베이스 스캔 테스트가 요구한다.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ripgrep \
    && rm -rf /var/lib/apt/lists/*

# 의존성 레이어: pyproject/uv.lock 변경 시에만 재설치된다.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

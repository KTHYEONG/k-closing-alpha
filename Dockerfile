FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1
# UV_NO_SYNC: 빌드시 --no-dev --frozen 으로 굳힌 venv를 uv run 이 런타임에
# 다시 동기화(=dev 의존성까지 재설치)하지 않도록 막는다. 실측: 이 변수 없이
# 컨테이너에서 uv run 을 실행하면 매번 mypy/ruff 등 14개 dev 패키지를
# 네트워크로 재설치했다.

# lightgbm 은 OpenMP 런타임(libgomp)을 요구한다 (VPS 실측: 누락 시 ImportError).
# ripgrep 은 test_kis_client_facade_removed.py 등 코드베이스 스캔 테스트가 요구한다.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ripgrep \
    && rm -rf /var/lib/apt/lists/*

# 의존성 레이어: pyproject/uv.lock 변경 시에만 재설치된다.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

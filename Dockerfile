FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /rl-debug-bench

COPY base/legacy_cleanrl/requirements.txt base/legacy_cleanrl/requirements.txt
RUN pip install --no-cache-dir -r base/legacy_cleanrl/requirements.txt

COPY pyproject.toml ./
COPY base/ base/
COPY tests/ tests/

RUN pip install --no-cache-dir pytest

CMD ["python", "base/legacy_cleanrl/ppo_cartpole.py"]

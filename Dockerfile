FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /rl-debug-bench

COPY base/requirements.txt base/requirements.txt
RUN pip install --no-cache-dir -r base/requirements.txt

COPY pyproject.toml ./
COPY base/ base/
COPY tests/ tests/

RUN pip install --no-cache-dir pytest

CMD ["python", "base/ppo_cartpole.py"]

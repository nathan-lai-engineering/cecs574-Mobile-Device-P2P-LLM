# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

# libgomp1: OpenMP runtime needed by numpy and onnxruntime (not bundled in their wheels)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y libgomp1

WORKDIR /app

# CPU-only torch wheel is ~200MB vs ~1.5GB for the default CUDA-enabled build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.0.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.coordinator.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.coordinator.txt

# Patch LLaMA for GQA support + torch.fx symbolic tracing compatibility
COPY patch_llama.py .
RUN python patch_llama.py

ENTRYPOINT ["/bin/bash"]

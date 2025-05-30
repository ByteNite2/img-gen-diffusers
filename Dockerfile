# syntax=docker/dockerfile:1.6
# Builds an image with FLUX 1 [schnell] fully pre-cached.
# ► Needs ≈ 65 GB RAM while building, so use a high-memory BuildKit worker.

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# ───────────────────────────── system packages ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ───────────────────────────── python packages ────────────────────────────────
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        "xformers==0.0.25.post1" \
        "diffusers==0.32.2" \
        "transformers==4.46.1" \
        "accelerate>=0.31.2,<2.0" \
        "huggingface_hub>=0.27.0,<1.0" \
        "peft>=0.10.0" \
        "sentencepiece>=0.1.99" \
        "protobuf>=3.20.3,<4" \
        psutil

# ───────────────────────────── model preload ──────────────────────────────────
ENV HF_HOME=/models
RUN mkdir -p /models

# Mount the Hugging Face token as a **BuildKit secret** named hf_token
RUN python - <<'PY'
import os, torch
from diffusers import FluxPipeline
FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    cache_dir="/models",
    torch_dtype=torch.bfloat16,
    token=os.getenv("HF_TOKEN")
).save_pretrained("/models/FLUX.1-schnell")
PY

WORKDIR /app
CMD ["/bin/bash"]
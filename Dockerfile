FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV HF_HOME=/models
RUN mkdir -p /models && mkdir -p /app
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    diffusers==0.27.2 transformers==4.40.0 accelerate==0.27.2 \
    huggingface_hub==0.20.2 xformers psutil

ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Download model safely (no class resolution)
RUN python3 -c "\
from huggingface_hub import login, snapshot_download;\
login(token='$HUGGINGFACE_HUB_TOKEN');\
snapshot_download(repo_id='black-forest-labs/FLUX.1-schnell', cache_dir='/models', local_dir='/models/flux', local_dir_use_symlinks=False)\
"

# Replace with working model_index.json
COPY model_index.json /models/flux/model_index.json

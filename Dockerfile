# ---- Base Serverless PyTorch image (GPU + CUDA 12.1 + Ubuntu 24.04) ----
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Workdir padrão
WORKDIR /workspace

# ---- Mapeia /workspace -> /runpod-volume (Network Volume) ----
# Compatibiliza Pod e Serverless
RUN rm -rf /workspace && ln -s /runpod-volume /workspace

# ---- Dependências ----
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Dependências essenciais para modelos HF
RUN python3 -m pip install \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    runpod \
    torch \
    einops \
    huggingface-hub

# Remove cache para reduzir tamanho
RUN python3 -m pip cache purge

# ---- Copia handler ----
COPY handler.py /workspace/handler.py

# ---- Start do handler ----
CMD ["python3", "-u", "/workspace/handler.py"]

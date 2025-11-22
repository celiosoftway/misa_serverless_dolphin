# RunPod Serverless Dockerfile para modelos HuggingFace (Phi / LoRA / Dolphin)
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Diretório principal
WORKDIR /workspace

# Dependências básicas
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN python3 -m pip install --upgrade pip

# Instalar libs essenciais para modelos HF
RUN python3 -m pip install transformers accelerate safetensors
RUN python3 -m pip install runpod
RUN python3 -m pip install hf-transfer

# Limpar cache pip
RUN python3 -m pip cache purge

# Copiar handler
COPY handler.py /workspace/handler.py

# Definir o cache HF no volume persistente
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume

# Comando de execução
CMD ["python3", "-u", "/workspace/handler.py"]

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install git+https://github.com/huggingface/diffusers
RUN python3 -m pip install transformers accelerate safetensors
RUN python3 -m pip install hf-transfer
RUN python3 -m pip install pillow
RUN python3 -m pip install runpod
RUN python3 -m pip cache purge

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]

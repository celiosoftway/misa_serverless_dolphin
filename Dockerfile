# --- Base image oficial do Runpod para Serverless GPU ---
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8-devel

# Evita interação
ENV DEBIAN_FRONTEND=noninteractive

# Instalações necessárias
RUN apt-get update && apt-get install -y git && apt-get clean

# Cria diretório da aplicação
WORKDIR /app

# Copia os arquivos do repositório para dentro da imagem
COPY handler.py /app/handler.py
COPY requirements.txt /app/requirements.txt
COPY runtime.yaml /app/runtime.yaml
COPY readme.md /app/readme.md

# Instala dependências Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Define o entrypoint do Runpod Serverless
CMD ["python", "-u", "handler.py"]

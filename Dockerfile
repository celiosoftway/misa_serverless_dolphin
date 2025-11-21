FROM runpod/serverless:gpu-cuda12.1

# Evita interação
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências opcionais
RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

COPY handler.py /app/handler.py
COPY requirements.txt /app/requirements.txt
COPY runtime.yaml /app/runtime.yaml
COPY readme.md /app/readme.md

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "-u", "handler.py"]

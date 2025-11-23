import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import runpod

# Se você mantiver o symlink workspace -> runpod-volume:
# MODEL_PATH = "/workspace/misa-dolphin"
# Se estiver acessando direto o mountpoint:
MODEL_PATH = "/runpod-volume/misa-dolphin"

tokenizer = None
model = None


def wait_for_path(path: str):
    print(f"⏳ Aguardando path existir: {path}")
    tries = 0
    while not os.path.exists(path):
        time.sleep(1)
        tries += 1
        if tries % 5 == 0:
            print(f"📌 Ainda aguardando path: {path} (tentativa {tries})")
    print(f"📂 Path disponível: {path}")


def debug_list(path: str):
    print("🔍 Listando conteúdo do diretório do modelo:")
    for root, dirs, files in os.walk(path):
        print(f"📁 DIR: {root}")
        for d in dirs:
            print(f"   📂 {d}/")
        for f in files:
            print(f"   📄 {f}")
    print("🔍 Fim da listagem.\n")


def load_model():
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return tokenizer, model

    # 1) Garante que o path existe
    wait_for_path(MODEL_PATH)

    # 2) Lista tudo que existe lá dentro
    debug_list(MODEL_PATH)

    # 3) Carrega tokenizer e modelo só de arquivo local
    print("🚀 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    print("🚀 Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()
    print("✅ Modelo carregado em modo de inferência.\n")

    return tokenizer, model


def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")

    if not prompt:
        return {"error": "Campo 'prompt' é obrigatório"}

    print(f"📝 Prompt recebido: {prompt[:80]}...\n")

    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
    )

    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return {"output": decoded}


runpod.serverless.start({"handler": handler})

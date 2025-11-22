import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/runpod-volume/misa-dolphin"

tokenizer = None
model = None

def wait_for_volume(path):
    print(f"⏳ Aguardando volume `{path}` montar...")
    while not os.path.exists(path):
        time.sleep(1)
    print("📁 Volume montado!")

def load_model():
    global tokenizer, model

    # Já carregado
    if tokenizer is not None and model is not None:
        return tokenizer, model

    # Aguarda montagem do volume
    wait_for_volume(MODEL_PATH)

    print("🚀 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    print("🚀 Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    # 👇 Aqui está o que você perguntou
    model.eval()
    print("✅ Modelo em modo de inferência (eval).")

    return tokenizer, model


def handler(job):
    """ job = { 'input': { 'prompt': 'texto...' } } """
    
    prompt = job["input"].get("prompt")
    if not prompt:
        return {"error": "Campo 'prompt' é obrigatório"}

    print(f"📝 Prompt recebido: {prompt[:80]}...")

    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"output": decoded}


runpod.serverless.start(
    {
        "handler": handler
    }
)

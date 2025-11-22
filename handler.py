import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Caminho do seu modelo dentro do storage persistente
MODEL_PATH = "/runpod-volume/misa-dolphin"   

# Lazy-load
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is not None:
        return tokenizer, model

    print(">> Carregando tokenizer/modelo a partir do disco local...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    model.eval()

    print("✅ Modelo Misa Dolphin carregado com sucesso!")
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

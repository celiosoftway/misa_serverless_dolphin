import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

# Caminho do modelo dentro do Volume do Runpod
MODEL_PATH = "/workspace/misa-phi"  
# MODEL_PATH = "/runpod-volume/misa-dolphin" 

print(">> Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(">> Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

def generate_text(prompt, max_tokens=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=1.1,
    )

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)


def handler(event):
    """
    event = {
        "input": "Texto do usuário",
        "max_tokens": 300,
        "temperature": 0.7
    }
    """
    prompt = event.get("input", "")
    if not prompt:
        return {"error": "input is required"}

    max_tokens = event.get("max_tokens", 256)
    temperature = event.get("temperature", 0.7)

    print(f">> Gerando resposta para: {prompt[:60]}...")

    output = generate_text(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return {"output": output}


runpod.serverless.start({"handler": handler})

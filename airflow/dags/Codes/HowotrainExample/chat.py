import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Reemplaza con tu token de Hugging Face
HUGGING_FACE_TOKEN = "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"

def authenticate():
    # Iniciar sesión en Hugging Face
    login(HUGGING_FACE_TOKEN)

def load_model_and_tokenizer(model_name):
    # Cargar el tokenizer y el modelo desde Hugging Face
    print(f"Cargando el modelo {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150, do_sample=True, top_p=0.95, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model_name = "SweatyCrayfish/llama-3-8b-quantized"
    
    # Autenticación en Hugging Face
    authenticate()

    # Cargar modelo y tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name)

    print("¡Modelo cargado exitosamente! Puedes empezar a hacer preguntas.")

    while True:
        # Leer entrada del usuario
        user_input = input("\nTú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Terminando la sesión de chat. ¡Adiós!")
            break

        # Generar respuesta
        response = generate_response(model, tokenizer, user_input)
        print(f"Modelo: {response}")

if __name__ == "__main__":
    main()



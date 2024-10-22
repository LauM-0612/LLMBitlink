import os
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import PeftModel, PeftConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

# Reemplaza con tu token de Hugging Face
HUGGING_FACE_TOKEN = "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"

# URL de la API de Django (ajusta la IP y puerto según tu configuración)
DJANGO_API_URL = "http://18.229.150.88:8800/get_all_data/"

app = FastAPI()

# Permitir orígenes específicos para evitar problemas de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por los dominios específicos si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    prompt: str

def authenticate():
    # Iniciar sesión en Hugging Face
    login(HUGGING_FACE_TOKEN)

def load_model_and_tokenizer(model_name):
    global model, tokenizer
    print(f"Cargando el modelo {model_name}...")

    # Cargar la configuración del modelo fine-tuned
    peft_config = PeftConfig.from_pretrained(model_name)

    # Cargar el tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # Cargar el modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Cargar el modelo fine-tuned
    model = PeftModel.from_pretrained(base_model, model_name)

def retrieve_data_from_django():
    try:
        # Hacer una solicitud GET a la API Django para obtener los datos
        response = requests.get(DJANGO_API_URL)
        response.raise_for_status()  # Verifica si la respuesta fue exitosa
        return response.json()
    except requests.RequestException as e:
        print(f"Error al obtener datos de la API Django: {e}")
        return None

def retrieve_relevant_knowledge(prompt, django_data):
    # Buscar información relevante dentro de los datos obtenidos de la API Django
    if django_data is None:
        return "No se pudo obtener información de la API Django."

    # Unir datos de las tres categorías (conversations, chunks, properties)
    knowledge_base = django_data.get("conversations", []) + django_data.get("chunks", []) + django_data.get("properties", [])

    # Ejemplo de lógica simple de recuperación (puedes mejorar esto)
    for entry in knowledge_base:
        if "input" in entry and entry["input"] in prompt:
            return entry["output"]
        elif "content" in entry and entry["content"] in prompt:
            return entry["content"]
        elif "description" in entry and entry["description"] in prompt:
            return entry["description"]
    
    return "No se encontró información relevante."

def generate_response(model, tokenizer, prompt, retrieved_info):
    # Formatear el prompt con la información recuperada
    formatted_prompt = f"### Human: {prompt}\n\n### Relevant Info: {retrieved_info}\n\n### Assistant:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al inicio (startup)
    authenticate()
    model_name = "pspedroelias96/LLMBitlink_Final"  # Usa el nombre de tu modelo fine-tuned
    load_model_and_tokenizer(model_name)
    print("¡Modelo cargado exitosamente!")
    
    yield  # Pausa para que la app siga corriendo

    # Código que se ejecuta al final (shutdown)
    print("Apagando la aplicación...")

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate_text(request_body: RequestBody):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado")
    
    prompt = request_body.prompt

    # Obtener datos de la API Django
    django_data = retrieve_data_from_django()

    # Recuperar información relevante para el prompt (RAG)
    retrieved_info = retrieve_relevant_knowledge(prompt, django_data)
    
    # Generar respuesta con la información recuperada
    response = generate_response(model, tokenizer, prompt, retrieved_info)
    
    return {"response": response}

if __name__ == "__main__":
    port = int(os.getenv("FASTAPI_PORT", 8800))
    uvicorn.run("ApiModel:app", host="0.0.0.0", port=port, reload=True)

import os
import torch
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

# Datos simulados de "documentos" o respuestas previas para RAG
knowledge_base = [
    {
        "user": "Pablo",
        "input": "¿Qué hace atractiva a la zona de La Perla para invertir?",
        "output": "La Perla es una de las zonas más tradicionales y demandadas de Mar del Plata, famosa por su proximidad a las playas y su ambiente residencial tranquilo. Las propiedades en esta área suelen tener una alta demanda tanto para alquileres turísticos como para vivienda permanente. Además, el crecimiento continuo de la infraestructura en la zona ha permitido una constante revalorización de las propiedades.",
        "answer_quality": 9.5,
        "answer_new": "Explica claramente por qué es una zona sólida para inversión inmobiliaria."
    },
    {
        "user": "Pablo",
        "input": "¿Cuáles son las ventajas de invertir en la zona de Güemes?",
        "output": "Güemes es una de las zonas más vibrantes y exclusivas de Mar del Plata, con una alta demanda tanto para vivienda permanente como para alquiler. Su cercanía a centros comerciales, tiendas de lujo y restaurantes de primer nivel la convierte en una opción muy codiciada. Además, la constante revalorización de las propiedades en esta zona garantiza una inversión segura y con alto retorno",
        "answer_quality": 9.5,
        "answer_new": "Enfocada en inversores que buscan alta demanda y exclusividad."
    }
]

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

def retrieve_relevant_knowledge(prompt):
    # Recuperar un fragmento relevante de la knowledge_base basado en el prompt
    for entry in knowledge_base:
        if entry["input"] in prompt:
            return entry["output"]
    return ""

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
    # Recuperar información relevante para el prompt (RAG)
    retrieved_info = retrieve_relevant_knowledge(prompt)
    
    # Generar respuesta con la información recuperada
    response = generate_response(model, tokenizer, prompt, retrieved_info)
    
    return {"response": response}

if __name__ == "__main__":
    port = int(os.getenv("FASTAPI_PORT", 8800))
    uvicorn.run("ApiModel:app", host="0.0.0.0", port=port, reload=True)

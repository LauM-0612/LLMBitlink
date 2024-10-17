import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por los dominios específicos si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

engine = 'gpt-3.5-turbo'

class ChatMessage(BaseModel):
    user_input: str

# Prompt template (unchanged)
prompt_template = """
# Rol
Eres un experto en ventas inmobiliarias llamado Max. Eres conocido por comunicar con precisión y persuasión la información sobre propiedades y servicios inmobiliarios. Tu estilo es amigable y accesible, mientras que tu enfoque es proactivo y orientado a soluciones, utilizando técnicas avanzadas de ventas y cierre.

# Objetivo
Proporcionar servicios de consultoría y asistencia de ventas de alto nivel a clientes y colegas. Debes demostrar competencia en técnicas avanzadas de ventas, negociación y gestión de relaciones con clientes, ofreciendo siempre una experiencia acogedora, profesional y confiable.

# Características de personalidad
* Amigable y accesible: Interactúa de forma cálida, creando una experiencia agradable.
* Profesional y confiable: Ofrece información precisa y actualizada.
* Proactivo y orientado a soluciones: Anticipa necesidades, ofreciendo soluciones innovadoras.
* Persuasivo pero respetuoso: Persuade usando datos y hechos, respetando siempre las preferencias del cliente.

# Habilidades clave a modelar
1. Comunicación efectiva: Simplifica la complejidad del mercado y describe las propiedades de forma clara.
2. Técnicas avanzadas de venta y cierre: Adapta las estrategias según el contexto y necesidades del cliente, incluyendo manejo de objeciones.
3. Análisis de datos del mercado inmobiliario: Analiza grandes volúmenes de datos para ofrecer recomendaciones detalladas sobre las tendencias del mercado.
4. Gestión eficiente de CRM: Utiliza herramientas CRM para automatizar y optimizar la gestión de relaciones, mejorando la eficiencia y la experiencia del cliente.

# Capacidades tecnológicas
* Integración con herramientas de realidad virtual para tours de propiedades, proporcionando una experiencia inmersiva y detallada.
* Uso de sistemas CRM avanzados para seguimiento y análisis de clientes, facilitando una gestión personalizada.
* Implementación de marketing digital para atraer y retener clientes mediante estrategias orientadas a datos.

# Ejemplos de interacción
- **Consulta inicial:** "Basado en tus preferencias y las tendencias actuales del mercado, te recomiendo considerar propiedades en estas áreas específicas..."
- **Negociación:** "Si estás interesado en hacer una oferta, puedo ayudarte a estructurarla de manera que sea atractiva para el vendedor y se ajuste a tu presupuesto. Además, puedo mostrarte cómo otros clientes han encontrado valor en propuestas similares."

# Notas
* Responde siempre en español latino.
* Sé persuasivo, específico y detallado, usando técnicas de venta como la escasez para crear urgencia.
* Responde únicamente a la consulta específica sin incluir información irrelevante.

Question: {question}  Context: {context}
"""

def get_completion(user_input: str) -> str:
    try:
        prompt = prompt_template.format(question=user_input, context="")
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in OpenAI API call")

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    logger.debug(f"Received message: {chat_message.user_input}")
    try:
        result = get_completion(chat_message.user_input)
        logger.debug(f"API response: {result}")
        return {"response": result}
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI OpenAI Integration!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
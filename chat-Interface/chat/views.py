import requests
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_protect, csrf_exempt  # Cambio de csrf_exempt a csrf_protect
from .models import Chunk  # Asegúrate de importar Chunk
import json
import logging
from django.contrib.auth.models import User  # Agrega esta línea
from django.contrib import messages
logger = logging.getLogger(__name__)
import requests
from django.http import JsonResponse
from .models import Conversation  # Importar el modelo de conversación
from django.contrib.auth.decorators import login_required

@login_required
def api_chat(request):
    if request.method == 'POST':
        try:
            # Leer el JSON enviado desde el frontend
            body_unicode = request.body.decode('utf-8')  # Decodificar el cuerpo
            body = json.loads(body_unicode)  # Convertir el cuerpo a un diccionario
            mensaje = body.get('mensaje')  # Obtener el valor del campo 'mensaje'

            # Verificar si el mensaje se recibió correctamente
            print(f"Mensaje enviado a FastAPI: {mensaje}")

            # Realizar la petición a la API de FastAPI
            url = 'http://localhost:8800/chat/'  # URL de la API de FastAPI
            headers = {'Content-Type': 'application/json'}
            payload = {'user_input': mensaje}

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                respuesta = data['response']

                # Guardar la conversación en la base de datos
                conversation = Conversation.objects.create(
                    user=request.user,  # Usuario autenticado
                    input=mensaje,
                    output=respuesta
                )

                # Devolver la respuesta al frontend
                return JsonResponse({'mensaje': respuesta})
            else:
                return JsonResponse({'error': 'Error al comunicarse con la API de FastAPI'}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
def index(request):
    return render(request, 'index.html')

@csrf_protect  # Protección CSRF habilitada
def api_view(request):
    if request.method == "POST":
        user_input = request.POST.get('mensaje', '')
        try:
            response = requests.post('http://localhost:8800/chat/', json={'user_input': user_input})
            if response.status_code == 200:
                response_data = response.json()
                if 'response' in response_data:
                    return JsonResponse({'mensaje': response_data['response']})
                else:
                    logger.error('Respuesta inesperada desde FastAPI: %s', response_data)
                    return JsonResponse({'error': 'Respuesta inesperada desde el servicio de chat'}, status=500)
            else:
                logger.error('Error de estado desde FastAPI: %s', response.status_code)
                return JsonResponse({'error': 'Error con el servicio de chat'}, status=response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error('Excepción al conectar con FastAPI: %s', e)
            return JsonResponse({'error': 'Error de conexión con el servicio de chat'}, status=500)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)


@csrf_protect
def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']

        if User.objects.filter(username=username).exists():
            messages.error(request, 'El nombre de usuario ya existe.')
            return redirect('register')
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            messages.success(request, 'Usuario registrado exitosamente.')
            return redirect('login')
    return render(request, 'register.html')


@csrf_protect
def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, 'Inicio de sesión exitoso.')
            return redirect('index')  # Redirige a la página principal o la que prefieras
        else:
            messages.error(request, 'Usuario o contraseña incorrecta.')
            return redirect('login')
    return render(request, 'login.html')


def user_logout(request):
    logout(request)
    messages.success(request, 'Sesión cerrada exitosamente.')
    return redirect('login')

@csrf_exempt
def save_vectorization(request):
    if request.method == 'POST':
        try:
            logger.info(f"Request body: {request.body}")
            data = json.loads(request.body)
            logger.info(f"Data received: {data}")
            for item in data:
                Chunk.objects.create(
                    document_id=item['document_id'],
                    content=item['content'],
                    embedding=item['embedding']
                )
            return JsonResponse({"status": "success"})
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return JsonResponse({"status": "fail", "error": str(e)}, status=400)
    return JsonResponse({"status": "fail"}, status=400)

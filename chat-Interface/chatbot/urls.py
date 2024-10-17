"""
URL configuration for chatbot project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from chat import views
# from chat.views import save_vectorization

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    # Asegúrate de que esta ruta acepta peticiones POST y no requiere el parámetro en la URL
    # path('api/', views.api_view, name='api'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('api/', views.api_chat, name='api_chat'),
    path('save_vectorization/', views.save_vectorization, name='save_vectorization'),  # Nueva URL
    path('send_data_to_fastapi/', views.send_data_to_fastapi, name='send_data_to_fastapi'),
    path('get_all_data/', views.get_all_data, name='get_all_data'),
]
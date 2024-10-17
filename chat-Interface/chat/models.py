from django.db import models
from django.contrib.auth.models import User

# Modelo para el historial de conversaciones
class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    input = models.TextField()
    output = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

class Chunk(models.Model):
    document_id = models.IntegerField()
    content = models.TextField()
    embedding = models.JSONField()

    def __str__(self):
        return f"Document {self.document_id}: {self.content[:50]}"

# Modelo para almacenar los inmuebles
class Property(models.Model):
    PROPERTY_TYPE_CHOICES = [
        ('house', 'Casa'),
        ('cabin', 'Cabaña'),
        ('apartment', 'Apartamento'),
        ('studio', 'Estudio'),
        ('land', 'Terreno'),
    ]
    location = models.CharField(max_length=255)  # Ubicación del inmueble
    price = models.DecimalField(max_digits=10, decimal_places=2)  # Precio del inmueble
    square_meters = models.DecimalField(max_digits=6, decimal_places=2)  # Metros cuadrados del inmueble
    property_type = models.CharField(max_length=20, choices=PROPERTY_TYPE_CHOICES)  # Tipo de inmueble
    description = models.TextField(blank=True, null=True)  # Descripción opcional del inmueble
    created_at = models.DateTimeField(auto_now_add=True)  # Fecha de creación

    def __str__(self):
        return f"{self.property_type.capitalize()} en {self.location} - {self.price} USD"

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PostgreSQL and other necessary packages
# Also remove the cache of apt-get to reduce the layer size
RUN apt-get update && \
    apt-get install -y libpq-dev gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file, to cache the pip install step
COPY ./requirements.txt .

# Install Python dependencies and upgrade pip
# Splitting into two RUN commands prevents re-running pip install when only pip is updated
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the Django project into the container
COPY . .

# Verify Python package installation
RUN pip list

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port 8800 to be accessible from the host
EXPOSE 8800

# Set environment variables
ENV NAME=World

# Command to run the application using Gunicorn with an increased timeout
CMD ["gunicorn", "--timeout", "120", "--bind", "0.0.0.0:8800", "chatbot.wsgi:application"]


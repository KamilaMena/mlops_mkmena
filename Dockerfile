# 1. Imagen base con Python
FROM python:3.14-slim

# 2. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Evitar archivos temporales de Python y mostrar logs inmediatamente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copiar e instalar las dependencias
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 5. Copiar los archivos del pipeline necesarios para la API y el modelo
COPY mlops_pipeline/ ./mlops_pipeline/

# 6. Copiar el modelo entrenado
COPY outputs/models/mejor_modelo.joblib outputs/models/mejor_modelo.joblib

# 7. Puerto que utilizará la API
EXPOSE 8000

# 8. Iniciar el servidor FastAPI
CMD ["uvicorn", "mlops_pipeline.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
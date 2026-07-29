FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Render (and most PaaS platforms) assign the actual listen port via $PORT
# and route to that -- a hardcoded --port here causes a 502 (platform can't
# reach the container) whenever Render's assigned port differs. Falls back
# to 8000 for local `docker run`/docker-compose, where $PORT is unset.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

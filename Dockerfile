FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

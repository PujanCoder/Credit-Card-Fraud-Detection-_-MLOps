FROM python:3.10-slim

WORKDIR /app

COPY flask_app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV FRAUD_THRESHOLD=0.50

EXPOSE 5001

CMD ["python", "-m", "flask_app.app"]
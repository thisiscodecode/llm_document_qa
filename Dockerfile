FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn==26.2.0 whitenoise==6.12.0

COPY . .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/media /app/chroma_db /app/staticfiles \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 120"]

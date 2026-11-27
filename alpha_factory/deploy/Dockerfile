FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY alpha_factory/requirements.txt alpha_factory/requirements.txt
RUN pip install --no-cache-dir -r alpha_factory/requirements.txt

COPY alpha_factory alpha_factory
RUN mkdir -p alpha_factory/data/logs alpha_factory/data/pnl alpha_factory/data/backups

CMD ["python", "-m", "alpha_factory", "run", "--config", "alpha_factory/config/settings.yaml"]

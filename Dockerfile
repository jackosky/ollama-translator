FROM python:3.11-slim@sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf

WORKDIR /app

RUN pip install poetry==2.1.2 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY ollama_translator/ ./ollama_translator/

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=5000

EXPOSE 5000

RUN adduser --disabled-password --no-create-home appuser
USER appuser

CMD ["sh", "-c", "python -m uvicorn ollama_translator.web.server:app --host $SERVER_HOST --port $SERVER_PORT"]

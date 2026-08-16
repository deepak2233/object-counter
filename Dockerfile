FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --quiet uv==0.11.33
COPY pyproject.toml uv.lock README.md ./
COPY counter ./counter
ARG UV_SYNC_ARGS=
RUN uv sync --frozen --no-dev --no-editable ${UV_SYNC_ARGS}


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    COUNTER_ENV=prod \
    COUNTER_LOG_FORMAT=json

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 counter

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY counter/resources/models.yaml counter/resources/models.dev.yaml ./config/
COPY resources ./resources
RUN mkdir -p /app/var/models && chown -R counter:counter /app

USER counter
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/readyz').read()"

CMD ["uvicorn", "counter.entrypoints.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "5000"]

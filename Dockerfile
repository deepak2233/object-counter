# Build and runtime are separate stages so the image that ships does not carry a
# compiler, the build cache, or the dev dependencies.
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies resolve from pyproject alone, so this layer is cached until the
# dependency list actually changes.
COPY pyproject.toml README.md ./
COPY counter ./counter
RUN pip install --quiet .


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    COUNTER_ENV=prod \
    COUNTER_LOG_FORMAT=json

# Runs as a non-root user: a service that decodes untrusted images should not be
# root inside its own container.
RUN useradd --create-home --uid 10001 counter

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# The application itself comes from the installed wheel in /opt/venv. Only the
# things that are not importable code are copied in: migrations, and the model
# catalogs at a stable path so a deployment can mount its own over them.
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY counter/resources/models.yaml counter/resources/models.dev.yaml ./config/
COPY resources ./resources

USER counter
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz').read()"

# Worker count is set by the deployment, not baked in: the right number depends
# on the CPU limit of the pod, and inference is CPU-bound. The app installs its
# own JSON logging over uvicorn's at startup.
CMD ["uvicorn", "counter.entrypoints.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "5000"]

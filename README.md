# Object counter

HTTP service for object detection and accumulated counts by class. It supports
TensorFlow Serving, ONNX Runtime, and TorchScript behind one detector interface.

## Local development

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Make.

```bash
make run
```

This starts the API on port 5000 with the development catalog, a fake detector,
and in-memory counts.

```bash
curl -F "threshold=0.9" \
  -F "file=@resources/images/cat.jpg" \
  http://localhost:5000/object-count
```

## Production profile

The packaged production catalog uses PostgreSQL and the `rfcn` model in
TensorFlow Serving.

```bash
make setup
make tfs-up
make run-prod
```

`make setup` installs the locked dependencies, starts PostgreSQL, and applies
the migrations. `make tfs-up` downloads and starts the public RFCN model used by
the exercise.

`make docker-up` starts the API and PostgreSQL with the development catalog. To
use TensorFlow Serving in the container stack:

```bash
make tfs-up
COUNTER_ENV=prod COUNTER_MODEL_CATALOG=/app/config/models.yaml make docker-up
```

## API

| Method and path | Result |
| --- | --- |
| `POST /object-detect` | Predictions above the requested threshold |
| `POST /object-count` | Current counts and accumulated totals |
| `GET /models` | Catalog models and load state |
| `GET /healthz` | Process liveness |
| `GET /readyz` | Database and default-detector readiness |
| `GET /docs` | OpenAPI UI |

Both POST endpoints accept multipart fields:

- `file`: required image
- `threshold`: optional number from 0 to 1
- `model_name`: optional catalog model name

Responses include the model name and version. Accumulated counts are isolated by
model name and version.

Errors use one envelope and return the request ID in both the body and the
`X-Request-ID` header:

```json
{
  "error": {
    "type": "invalid_threshold",
    "message": "threshold must be between 0.0 and 1.0, got 90.0",
    "request_id": "7c5f6f4eb77348b3b4e52da19a738c31"
  }
}
```

## Configuration

Copy `.env.example` to `.env` for local overrides. Common settings are:

| Variable | Purpose |
| --- | --- |
| `COUNTER_ENV` | `dev`, `test`, or `prod` profile |
| `COUNTER_PERSISTENCE` | `memory` or `sql`; production requires `sql` |
| `COUNTER_DATABASE_URL` | SQLAlchemy database URL |
| `COUNTER_MODEL_CATALOG` | YAML or JSON model catalog |
| `COUNTER_TFS_BASE_URL` | TensorFlow Serving base URL |
| `COUNTER_DEFAULT_THRESHOLD` | Threshold used when omitted |
| `COUNTER_MAX_IMAGE_BYTES` | Upload byte limit |
| `COUNTER_MAX_IMAGE_SIDE` | Maximum TFS input side |

Dependencies are resolved in `uv.lock`. Optional runtime groups are `onnx`,
`torch`, and `s3`.

## Verification

```bash
make test
make lint
make typecheck
make coverage
```

Database tests use PostgreSQL when `TEST_DATABASE_URL` is set and SQLite
otherwise. CI uses PostgreSQL and also builds the container image.

## Structure

```text
counter/
  domain/       value objects, ports, and use cases
  adapters/     detector and repository implementations
  entrypoints/  FastAPI and CLI
  bootstrap.py  dependency construction
migrations/     Alembic migrations
tests/          unit, integration, and end-to-end tests
```

## Assignment map

| Task | Delivery |
| --- | --- |
| Prediction endpoint | `POST /object-detect` |
| Relational repository | `counter/adapters/repo/sql.py` and `migrations/` |
| Review and fixes | [Code review](docs/CODE_REVIEW.md) |
| Internal models | [Multi-model setup](docs/MULTI_MODEL.md) |
| Integration and end-to-end tests | [Testing](docs/TESTING.md) |
| Framework extension | [Framework adapters](docs/MULTI_FRAMEWORK.md) |

The framework adapters demonstrate the extension path. A deployment should
still run contract tests with its actual model files before release.

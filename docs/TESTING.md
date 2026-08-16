# Testing

## Test levels

| Level | Command | Coverage |
| --- | --- | --- |
| Unit | `make test-unit` | Domain logic, parsing, preprocessing, adapters through stubs |
| Integration | `make test-integration` | SQL repository and Alembic migrations |
| End to end | `make test-e2e` | FastAPI routes, middleware, schemas, errors, and persistence flow |

`make test` runs all levels. `make coverage` adds a line and branch coverage
report.

## Database matrix

When `TEST_DATABASE_URL` is unset, database tests use a temporary SQLite file.
This keeps local verification self-contained.

CI sets `TEST_DATABASE_URL` to PostgreSQL. The PostgreSQL run is required for
the concurrent increment test and the production upsert dialect.

```bash
make db-up
make test-integration
```

Migrations are applied with Alembic in tests. The suite checks upgrade,
downgrade, and preservation of pre-`0002` count rows.

## Important regressions covered

- concurrent increments do not lose counts
- totals are isolated by model name and version
- thresholds reject invalid values
- unknown models return 404
- corrupt, oversized, and high-pixel-count images are rejected
- early 413 responses include the normal request ID envelope
- TFS timeouts, HTTP failures, malformed data, and unavailable status map to 503
- class-aware NMS keeps overlapping objects from different classes
- ONNX and TorchScript receive their declared input layout
- repository errors do not expose driver detail
- readiness fails when the database or default detector is unavailable

## Runtime adapter tests

ONNX and TorchScript tests use injected sessions or modules. TensorFlow Serving
tests use `httpx.MockTransport`. These tests are fast and deterministic, but
they are not model-release tests.

A release pipeline should add a contract test for every production model using
the real artifact and runtime. That test should verify preprocessing, labels,
representative predictions, and resource limits.

## CI

The GitHub Actions job:

1. installs the exact `uv` version
2. syncs from `uv.lock`
3. runs Ruff
4. runs mypy
5. runs tests with coverage against PostgreSQL
6. builds the Docker image

The repository does not claim load-test, GPU, or live TensorFlow Serving
coverage.

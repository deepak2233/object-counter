# Object counter

Send it an image and a confidence threshold. It detects the objects, counts the
ones scoring above the threshold, and groups them by class.

## Run it

You need Python 3.11+ and Make. Docker only if you want the database.

```bash
make run
```

That serves on port 5000 with a fake detector and in-memory counts. Nothing to
download first.

```bash
curl -F "threshold=0.9" -F "file=@resources/images/cat.jpg" localhost:5000/object-count
curl -F "threshold=0.9" -F "file=@resources/images/cat.jpg" localhost:5000/object-detect
```

With Postgres behind it, so the counts survive a restart:

```bash
make setup      # venv, database, migrations
make run-prod
```

Or `make docker-up` for the whole stack in containers.

`make help` lists the rest.

## Endpoints

| Path | What you get |
| --- | --- |
| `POST /object-detect` | The predictions |
| `POST /object-count` | Counts by class for this image, plus the running totals |
| `GET /models` | Which models this instance can serve |
| `GET /healthz` `GET /readyz` | Liveness and readiness |
| `GET /docs` | OpenAPI UI |

Both POSTs take the same form: `file`, and optionally `threshold` (0 to 1) and
`model_name`. Boxes come back normalised to 0..1 with the origin top-left, so
scale them to whatever you are drawing on.

Errors all look the same, and the id is in the `X-Request-ID` header too:

```json
{"error": {"type": "invalid_threshold", "message": "threshold must be between 0.0 and 1.0, got 90.0", "request_id": "9f2c…"}}
```

404 for an unknown model, 413 too big, 415 not an image, 422 a bad threshold,
503 when the detector or the database is down.

There is a CLI over the same code: `make cli IMAGE=resources/images/food.jpg`.

## Configuration

`.env.example` lists every setting. The ones you will actually touch:

| Variable | Default | |
| --- | --- | --- |
| `COUNTER_ENV` | `dev` | `dev` is fakes and memory, `prod` is real models and Postgres |
| `COUNTER_DATABASE_URL` | local Postgres | Any SQLAlchemy URL |
| `COUNTER_MODEL_CATALOG` | packaged | Your own catalog of models |
| `COUNTER_TFS_BASE_URL` | `localhost:8501` | TensorFlow Serving |
| `COUNTER_DEFAULT_THRESHOLD` | `0.5` | Used when the request omits one |

A bad value stops the process at startup and names the field.

## Tests

```bash
make test          # all 165
make test-unit     # no I/O
make coverage
make lint typecheck
```

91% coverage. The database tests run on Postgres when `TEST_DATABASE_URL` points
at one and on SQLite when it does not. So the suite works on a laptop with
nothing installed, and CI still runs it against the dialect that ships.

## Layout

```
counter/
  domain/       models, ports, counting logic. No I/O, no frameworks
  adapters/
    detector/   TF Serving, ONNX Runtime, TorchScript, fake
    repo/       in-memory and SQL
  entrypoints/  FastAPI app and CLI
  config.py     typed settings
  bootstrap.py  the only file that picks which adapter is used
migrations/     Alembic
tests/          unit, integration, e2e
```

Nothing in `domain/` imports from `adapters/` or `entrypoints/`.

## Serving your own models

A model is a YAML catalog entry, not code. Point `COUNTER_MODEL_CATALOG` at your
file, put the weights in your model store, and the name works on every endpoint.
Artifacts can be pinned to a SHA-256, so you know what you are actually serving.
`config/models.example.yaml` has four models across three frameworks.

## Docs

- [Code review](docs/CODE_REVIEW.md) of the original code: 19 findings, all fixed
- [Architecture](docs/ARCHITECTURE.md), and the options that were rejected
- [Internal models](docs/MULTI_MODEL.md) and [frameworks](docs/MULTI_FRAMEWORK.md)
- [Testing](docs/TESTING.md)
- [Decision records](docs/adr)

## The exercise

| # | Task | Where |
| --- | --- | --- |
| 1 | Endpoint returning predictions | `POST /object-detect` |
| 2 | Relational `ObjectCountRepo` | `counter/adapters/repo/sql.py`, `migrations/` |
| 3, 4 | Review and fixes | [CODE_REVIEW.md](docs/CODE_REVIEW.md) |
| 5 | Internally trained models | [MULTI_MODEL.md](docs/MULTI_MODEL.md) |
| 6a | Integration and e2e tests | [TESTING.md](docs/TESTING.md) |
| 6b | Several frameworks | [MULTI_FRAMEWORK.md](docs/MULTI_FRAMEWORK.md) |

Both halves of task 6 are done.

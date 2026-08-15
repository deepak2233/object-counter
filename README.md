# Object counter

Detects objects in an image and counts the ones scoring above a threshold,
grouped by class. Rewrite of the [NIQ object-counter
exercise](https://github.com/arenan02/object-counter), keeping its hexagonal
architecture and its `/object-count` contract, with the parts that would not
survive production replaced.

Solutions to the six assignment tasks are indexed in [what the assignment
asked](#what-the-assignment-asked).

## Quickstart

Requirements: Python 3.11+, Docker (only for the database and TensorFlow
Serving), GNU Make.

```bash
make run                      # API on http://localhost:5000, fake detector, in-memory counts
```

That needs no database, no weights and no model server. To hit it:

```bash
curl -F "threshold=0.9" -F "file=@resources/images/cat.jpg" localhost:5000/object-count
curl -F "threshold=0.9" -F "file=@resources/images/cat.jpg" localhost:5000/object-detect
```

For the real thing, with Postgres, migrations applied and counts that survive a
restart:

```bash
make setup                    # virtualenv + Postgres container + migrations
make run-prod
```

Or the whole stack in containers:

```bash
make docker-up                # builds the image, waits for Postgres, migrates, starts the API
```

`make help` lists every target. Anything the project can do has one.

Interactive API documentation is at http://localhost:5000/docs.

## The API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/object-detect` | Returns the predictions above the threshold |
| `POST` | `/object-count` | Returns counts by class for this image, plus running totals |
| `GET` | `/models` | Lists the models this instance can serve |
| `GET` | `/healthz` | Liveness. Checks nothing else |
| `GET` | `/readyz` | Readiness: store reachable and migrated, default model loadable |
| `GET` | `/docs` | OpenAPI UI |

Both detection endpoints take the same multipart form: `file` (required),
`threshold` (optional, defaults to `COUNTER_DEFAULT_THRESHOLD`, must be in
`[0, 1]`), and `model_name` (optional, defaults to the catalog default).

`POST /object-detect`:

```json
{
  "model": "fake",
  "threshold": 0.7,
  "count": 2,
  "predictions": [
    {
      "class_name": "cat",
      "score": 0.999190748,
      "box": {"xmin": 0.367288858, "ymin": 0.278333426, "xmax": 0.735821366, "ymax": 0.6988855}
    },
    {
      "class_name": "cat",
      "score": 0.752194285,
      "box": {"xmin": 0.101288858, "ymin": 0.118333426, "xmax": 0.335821366, "ymax": 0.4988855}
    }
  ]
}
```

Boxes are normalised to `0..1` with the origin top-left, so a client can scale
them to whatever resolution it displays.

`POST /object-count`:

```json
{
  "model": "fake",
  "threshold": 0.7,
  "current_objects": [{"object_class": "cat", "count": 2}],
  "current_total": 2,
  "total_objects": [{"object_class": "cat", "count": 6}],
  "accumulated_total": 6
}
```

`current_objects` and `total_objects` keep the names and shape the original
service used. `current_total` and `accumulated_total` are additions.

Errors share one envelope, and every one of them carries the request id that is
also in the `X-Request-ID` response header and in the logs:

```json
{"error": {"type": "invalid_threshold", "message": "threshold must be between 0.0 and 1.0, got 90.0", "request_id": "9f2c…"}}
```

| Status | When |
| --- | --- |
| 404 | Model name is not in the catalog |
| 413 | Upload above `COUNTER_MAX_IMAGE_BYTES` |
| 415 | Payload is not a decodable image |
| 422 | Threshold outside `[0, 1]`, or malformed form data |
| 503 | Detector or database unavailable |

There is also a CLI over the same use cases:

```bash
make cli IMAGE=resources/images/food.jpg THRESHOLD=0.6
.venv/bin/python -m counter.entrypoints.cli models
```

## Configuration

Every setting is typed and validated at startup in `counter/config.py`, read
from the environment with the `COUNTER_` prefix or from `.env`. Start from
`.env.example`.

| Variable | Default | Notes |
| --- | --- | --- |
| `COUNTER_ENV` | `dev` | `dev` uses fakes and memory; `prod` uses the catalog and SQL |
| `COUNTER_PERSISTENCE` | profile-dependent | `memory` or `sql` |
| `COUNTER_DATABASE_URL` | local Postgres | Any SQLAlchemy URL; Postgres, MySQL and SQLite have upserts |
| `COUNTER_MODEL_CATALOG` | packaged catalog | Path to your own catalog file |
| `COUNTER_DEFAULT_MODEL` | catalog default | Overrides the catalog's default |
| `COUNTER_PRELOAD_MODELS` | `false` | Load every model at boot instead of on first use |
| `COUNTER_TFS_BASE_URL` | `http://localhost:8501` | TensorFlow Serving endpoint |
| `COUNTER_MAX_IMAGE_BYTES` | `10485760` | Upload limit |
| `COUNTER_MAX_IMAGE_SIDE` | `1024` | Images are downscaled to this before serialisation to TFS |
| `COUNTER_DEFAULT_THRESHOLD` | `0.5` | Used when the request omits one |
| `COUNTER_LOG_FORMAT` | `json` | `text` when you are the one reading it |

An invalid value fails the process at startup with the field name in the
message, rather than at the first request.

## Running against TensorFlow Serving

```bash
make tfs-up                   # downloads the RFCN model (~600 MB) and starts TFS
make run-prod                 # the prod profile serves the rfcn catalog entry by default
curl -F "threshold=0.9" -F "file=@resources/images/boy.jpg" \
     -F "model_name=rfcn" localhost:5000/object-count
```

Serving privately trained models instead is a catalog entry and an artifact in
your model store; no code change. See [docs/MULTI_MODEL.md](docs/MULTI_MODEL.md).

## Tests

```bash
make test                # everything; database tests fall back to SQLite
make test-unit           # no I/O
make test-integration    # database and adapters, against Postgres
make test-e2e            # the HTTP stack
make coverage
```

165 tests, 91% line coverage. Unit tests need nothing installed. Integration and
end-to-end tests use Postgres when `TEST_DATABASE_URL` points at one and SQLite
otherwise, which keeps `make test` runnable on a laptop while CI runs the
dialect that ships. What each level is for is in
[docs/TESTING.md](docs/TESTING.md).

Static checks:

```bash
make lint typecheck      # ruff + ruff format + mypy (strict on the package)
make verify              # what CI runs
```

## Layout

```
counter/
  domain/            models, ports, pure prediction functions, use cases — no I/O, no frameworks
  adapters/
    detector/        TF Serving, ONNX Runtime, TorchScript, fake; catalog, registry, artifact store
    repo/            in-memory and SQL implementations of ObjectCountRepo
  entrypoints/
    api/             FastAPI app, routes, schemas, error mapping, middleware
    cli.py           same use cases from a terminal
  config.py          typed settings
  bootstrap.py       composition root: the only module that picks implementations
migrations/          Alembic; the schema ships as reviewable scripts
tests/unit|integration|e2e
docs/                architecture, code review, model and framework guides, ADRs
```

The dependency rule: `domain` imports nothing from `adapters` or `entrypoints`.
The original violated it. `domain/actions.py` imported PIL and a debug module
that wrote JPEGs to disk on every request.

## What the assignment asked

| # | Task | Where |
| --- | --- | --- |
| 1 | New endpoint returning predictions | `POST /object-detect` in `counter/entrypoints/api/routes.py`, `DetectObjects` in `counter/domain/actions.py` |
| 2 | Relational `ObjectCountRepo` | `counter/adapters/repo/sql.py`, `migrations/versions/0001_create_object_counts.py`, Postgres in `docker-compose.yml` |
| 3 | Review the code, propose improvements | [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md), 19 findings with severity |
| 4 | Implement at least one | All 19 are implemented; four further changes were considered and rejected, with reasons |
| 5 | Multiple internally trained models | [docs/MULTI_MODEL.md](docs/MULTI_MODEL.md): catalog, digest-pinned artifact store, `GET /models` |
| 6a | Integration and e2e tests | [docs/TESTING.md](docs/TESTING.md): 46 integration/e2e tests against real Postgres |
| 6b | Several deep learning frameworks | [docs/MULTI_FRAMEWORK.md](docs/MULTI_FRAMEWORK.md): TF Serving, ONNX Runtime and TorchScript behind one port |

Both options of task 6 are done rather than one.

The reasoning behind the four decisions worth arguing about (hexagonal
boundaries, FastAPI over Flask, the atomic upsert, the framework registry) is in
[docs/adr](docs/adr). Architecture alternatives that were considered and
rejected are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

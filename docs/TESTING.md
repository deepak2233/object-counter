# Tests

Assignment task 6a. 165 tests, 91% line coverage, 3.5 seconds for the whole
suite against Postgres.

| Level | Count | Command | What it proves |
| --- | --- | --- | --- |
| unit | 119 | `make test-unit` | Logic and adapter translation, no I/O |
| integration | 19 | `make test-integration` | The SQL adapter and the migrations against a real database |
| e2e | 27 | `make test-e2e` | The HTTP stack: routing, validation, status codes, persistence |

The original suite had five tests. Two covered pure functions, two covered the
counting use case with mocks, and one posted an image and asserted
`json.loads(response.data) != None`, which is true for `{}`, `[]` and `0`. There
was no test of any repository, no error path, and nothing that would have caught
the concurrency bug, the float class ids, the RGBA crash or the unvalidated
threshold.

## What each level is for

**Unit.** Everything that can be decided without a socket or a file. The
prediction functions, the use cases, the in-memory repository, the settings and
their profile defaults, catalog validation, checksum verification, image
decoding, and the three detector adapters driven through stubs.

The adapter tests are the ones worth arguing for. `parse_predictions` is a
module-level function, so the TF Serving wire format is tested with a dict —
including a truncated body and a `num_detections` that lies. The HTTP behaviour
is tested through `httpx.MockTransport`, so timeouts and 500s are ordinary test
cases rather than things you hope work. ONNX and TorchScript take an injected
session, so the tensor layout and the box arithmetic are checked without
installing either wheel.

**Integration.** The SQL adapter against a database whose schema was built by
`alembic upgrade head` — not `metadata.create_all()`, which would test a schema
nobody deploys and let a broken migration ship green. Both directions of the
migration run, because a migration nobody can roll back is a migration nobody
can deploy on a Friday.

The test that matters most fires 200 parallel increments at one class through a
thread pool and asserts the total is exactly 200. It is skipped on SQLite, which
serialises writers, so it only runs when `TEST_DATABASE_URL` points at Postgres
— and it is the reason the write is an upsert rather than a read-modify-write.

Failure modes are covered too: an unreachable database, a database that is
reachable but unmigrated, and a dialect with no upsert defined all raise
`RepositoryError` rather than leaking a driver exception.

**End to end.** The real FastAPI app through `TestClient`, with a fake detector
and a real database. Both endpoints, threshold behaviour at both extremes, the
agreement between `/object-detect` and `/object-count`, totals accumulating
across requests and across a simulated restart, every error status (404, 413,
415, 422, 503), the request id echo, and the OpenAPI document.

Only the model is fake, deliberately. Asserting on the output of a real detector
tests the weights, not the service, and it makes the suite depend on a 600 MB
download.

## Running against Postgres

```bash
make db-up                    # Postgres in Docker + the test database
make test-integration
```

Without `TEST_DATABASE_URL`, the database tests fall back to a temporary SQLite
file, so `make test` works on a machine with nothing installed. That is a real
trade-off: SQLite exercises the SQL path and the migrations but not the
PostgreSQL upsert or concurrent writers, so CI sets `TEST_DATABASE_URL` against
a Postgres service container and the concurrency test only counts there.

## Fixtures

`tests/conftest.py` builds the app through the same `create_app(services=...)`
seam the production code uses. No monkeypatching of module globals, no import
side effects — the fake detector and the throwaway database are constructor
arguments. That is the practical payoff of the composition root.

Image fixtures are generated rather than committed where their format is the
point: an RGBA PNG and a grayscale JPEG, because those two are what the original
decoder crashed on.

## What is not covered

- No load test. p99 under concurrency is a claim this repo does not make.
- No test that a real `.onnx` or `.pt` file loads. That needs the wheels and an
  artifact, and belongs in a nightly job with the model store attached.
- No contract test against a live TensorFlow Serving. The wire format is pinned
  by fixtures taken from the documented response shape; a real TFS in CI would
  cost a 600 MB pull per run.
- Mutation testing would be the honest next step for the domain, where the tests
  are cheap and the logic is the product.

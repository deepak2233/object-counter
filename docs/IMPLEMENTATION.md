# What was built, and why

A walkthrough of the work at code level: every task, the reasoning behind each
decision, and what it would actually take to replace any piece of it.

Numbers, so the scale is clear: 2,640 lines of production Python, 1,579 lines of
tests, 165 tests at 91% coverage, 3 seconds for the full suite.

## The tasks

### 1. An endpoint that returns predictions

`POST /object-detect` in `counter/entrypoints/api/routes.py:53`.

The interesting decision is not the route, it is that the route does almost
nothing. `DetectObjects` (`counter/domain/actions.py:14`) validates the
threshold, calls the detector, filters. `CountDetectedObjects` then wraps
*that same object* rather than calling the detector itself:

```python
class CountDetectedObjects:
    def __init__(self, object_detector, object_count_repo):
        self._detect_objects = DetectObjects(object_detector)
        self._object_count_repo = object_count_repo
```

So counting is defined as "count what detect returned". The two endpoints cannot
drift apart, and `test_counting_and_detecting_agree` pins it: the count equals
the number of predictions for the same image and threshold. Had the count path
called the detector separately, a later change to one filter would silently make
the endpoints disagree and nothing would fail.

Both routes are `async def` but push the work through `run_in_threadpool`.
Inference and psycopg both block. Running a forward pass on the event loop
stalls every other in-flight request for its whole duration.

### 2. A relational adapter for `ObjectCountRepo`

`counter/adapters/repo/sql.py`, 200 lines, plus the Alembic migration in
`migrations/versions/0001_create_object_counts.py`.

One table, `object_counts(object_class PK, count, updated_at)`, and one
statement per write:

```sql
INSERT INTO object_counts (object_class, count) VALUES (:object_class, :count)
ON CONFLICT (object_class)
DO UPDATE SET count = object_counts.count + excluded.count, updated_at = now()
```

Why an upsert rather than read-add-write: the original in-memory adapter did
`store[key] = ObjectCount(key, stored.count + new.count)`, which loses counts the
moment two requests overlap. Both threads read 4, both write 5. The database has
to do the addition, in one statement, under its own lock. `test_parallel_increments_do_not_lose_counts`
fires 200 concurrent increments at Postgres and asserts the total is exactly 200.

Three details that came out of writing it:

- The class name is the primary key, not a surrogate id. The uniqueness
  constraint is the thing that makes the upsert atomic; a surrogate key would
  need a separate unique index to get the same guarantee back.
- `_merge_duplicates` (`sql.py:145`) collapses repeated classes before the
  statement is built. A multi-row upsert naming the same key twice is a runtime
  error on PostgreSQL: "cannot affect row a second time".
- `_upsert_statement` (`sql.py:163`) dispatches on `engine.dialect.name`.
  PostgreSQL and SQLite use `ON CONFLICT`, MySQL uses `ON DUPLICATE KEY UPDATE`.
  An unknown dialect raises rather than silently falling back to something that
  looks like it works.

The repo takes an `Engine`, not a URL. Connection pooling belongs to the
composition root, and injecting it is what lets one adapter be pointed at
Postgres in CI and SQLite on a laptop.

### 3 and 4. Review, and fixing all of it

19 findings in `docs/CODE_REVIEW.md`, graded by what hurts first. All fixed. The
four that would page someone: no timeout on the model call, debug JPEGs written
from inside the domain on every request, `app.run(debug=True)` in the entrypoint
the README tells you to run in production, and the lost counts above.

The one worth reading at code level is the label map:

```python
self.classes_dict = {label['id']: label['display_name'] for label in labels}  # int keys
class_name = self.classes_dict[detection_class]                               # 17.0, a float
```

This survives by accident. `hash(17.0) == hash(17)` in Python, so the lookup
finds the int key. It stops surviving the moment a model returns `17.5` for a
padded slot or a serving version sends ids as strings. `class_name_for`
(`labels.py:61`) coerces through `int()` and falls back to `class_<id>` for
unknown ids. Falling back rather than dropping, because a silently discarded
detection corrupts the counts, and counts are the product.

### 5. Several internally trained models

A model became a config entry instead of a constructor call. Three pieces:

- `catalog.py` — a YAML file validated by pydantic at startup. A typo fails the
  process with a precise message instead of a `KeyError` at 3am.
- `registry.py` — resolves a name to a detector, builds it on first use, caches
  it under a lock with a double-checked read so two cold requests build one
  session rather than two.
- `artifacts.py` — resolves an artifact URI to a local path and verifies its
  SHA-256. Downloads land in a `.part` file and are renamed only after the digest
  matches, so a crashed download is never picked up as a valid cache entry.

Pinning by digest is the part that matters operationally. It is the difference
between "we serve retail-shelf 3.2.1" and "we serve whatever was in the bucket
this morning".

### 6a. Integration and e2e tests

165 tests in three levels. The original had five, one of which asserted
`json.loads(response.data) != None` — true for `{}`, `[]` and `0`.

The two techniques doing the heavy lifting:

**Injected transports and sessions.** `parse_predictions` is a module-level
function, so the TF Serving wire format is tested with a dict, including a
truncated body and a lying `num_detections`. HTTP behaviour goes through
`httpx.MockTransport`, so a timeout is an ordinary test case. `InferenceSession`
is a `Protocol` covering the two methods the ONNX adapter calls, so a nine-line
stub tests the tensor layout without installing a 200 MB wheel.

**Migrations, not `create_all()`.** The integration fixture runs
`alembic upgrade head` and rolls back to `base` afterwards. Testing against a
schema built by `metadata.create_all()` would prove nothing about the schema
that actually deploys, and would let a broken migration ship green.

### 6b. Several deep learning frameworks

Three backends behind one port: TF Serving over HTTP, ONNX Runtime in-process,
TorchScript in-process. The port is four lines, and the domain never sees a
tensor.

The design decision worth defending is `onnx_ops.py` — 213 lines of letterbox,
NMS and coordinate maths that import numpy and nothing else. That is where
detector bugs live: padding offsets, xywh versus xyxy, boxes that map back to the
wrong place. Keeping it free of onnxruntime means it is tested in milliseconds on
any machine, with no wheel installed.

`_is_channels_first` is a small example of the payoff. The YOLOv8
export is `(1, 4+nc, 8400)` and its transpose is equally common, so the layout is
identified from the label count when known and falls back to "boxes outnumber
channels" when not. That heuristic is a one-line function with its own tests
instead of a comment nobody checks.

## Replacing any of it

Every seam is a port in `counter/domain/ports.py`. What each swap actually costs:

| Replace | Implement | Register in | Cost |
| --- | --- | --- | --- |
| FastAPI with Flask, Litestar, gRPC | a new entrypoint | nothing else | ~370 lines under `entrypoints/api/`; domain untouched |
| Postgres with MySQL | nothing | already there | a dialect branch that already exists (`sql.py:189`) |
| Postgres with Mongo, DynamoDB, Redis | `ObjectCountRepo`, 2 methods | `bootstrap.build_repo` | ~40 lines, if the store has an atomic increment |
| TF Serving with Triton, OpenVINO, TensorRT | `ObjectDetector`, 2 members | `registry._build` + `Framework` literal | one adapter file; reuse `images.decode` and `onnx_ops` |
| S3 with GCS, Azure, MLflow | one `_fetch_*` method | `ArtifactStore.resolve` | ~25 lines |
| YAML catalog with a model registry service | `spec_for` and `models` | `bootstrap.build_registry` | the catalog is already an object, not a file read scattered around |
| JSON logs with OpenTelemetry | a handler | `configure_logging` | request id is already a ContextVar |

These are cheap because of one rule, and the rule is checkable:

```bash
grep -rE "^(from|import) (fastapi|sqlalchemy|httpx|onnxruntime|torch|PIL|pydantic)" counter/domain/
```

That returns nothing. The whole domain imports `abc`, `collections`,
`dataclasses`, `logging` and its own modules. Everything else sits behind an
interface the domain defines.

`bootstrap.py` is the only file that names concrete classes. Swapping an adapter
is an edit there and a new file — never a change to a use case, a route, or
another adapter.

### What is not cheap to replace, and why

Two assumptions are baked into the domain, and honesty about them is more useful
than pretending everything is pluggable:

**A prediction is one class, one score, one box.** `Prediction` in
`domain/models.py:27` is that shape. Segmentation masks, keypoints, or
multi-label detections do not fit, and adding them means changing the model, both
schemas, and every adapter. That is the right trade for a counting service; it
would be the wrong trade for a general vision platform.

**Counts are a running total with no time dimension.** `ObjectCountRepo` is
`read_values` and `update_values` over an aggregate. There is no per-request
history, so "how many cats last Tuesday" cannot be answered. The row was
incremented and the event thrown away. Adding it means an events table, a new
port method, and a decision about retention. Worth doing the moment anyone asks
a question with a date in it, and not before.

**Retries are not idempotent.** A client that retries a failed upload counts the
image twice. Fixing it properly needs a client-supplied idempotency key and a
dedupe table. The assignment has no requirement for it; it is the first thing I
would raise with whoever owns the product.

## What I would do next

In order: an events table so counts can be sliced by time. Per-model latency and
error metrics, labelled with model name and version, so a regression after a
model bump is visible without correlating deploys by hand. An import-linter
contract in CI, so the dependency rule is enforced by a machine rather than by
review. And a nightly job that loads a real ONNX artifact from the model store,
because CI proves the adapter is correct, not that any particular file loads.

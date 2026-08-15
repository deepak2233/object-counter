# Architecture

## The shape

Hexagonal, kept from the original because it was the right call there and is the
right call here: the interesting part of this service is that the model, the
store and the transport are all replaceable, and ports make that structural
rather than aspirational.

```
                    entrypoints (drivers)
        ┌───────────────────────────────────────────┐
        │  FastAPI routes            CLI            │
        └───────────────┬───────────────────────────┘
                        │  calls use cases, maps errors to status codes
        ┌───────────────▼───────────────────────────┐
        │  domain                                   │
        │    DetectObjects, CountDetectedObjects    │
        │    over_threshold, count  (pure)          │
        │    ports: ObjectDetector, ObjectCountRepo │
        │           ObjectDetectorRegistry          │
        └───────────────▲───────────────────────────┘
                        │  implemented by
        ┌───────────────┴───────────────────────────┐
        │  adapters (driven)                        │
        │    TF Serving · ONNX Runtime · TorchScript│
        │    fake · in-memory · SQLAlchemy          │
        └───────────────────────────────────────────┘
```

`counter/bootstrap.py` is the only module that knows which implementation is
plugged in. Everything else takes its collaborators through a constructor, which
is what lets the end-to-end tests run the real HTTP stack against a fake
detector and a throwaway database without patching a single module global.

Two rules are enforced by review rather than by a tool: `domain` imports nothing
from `adapters` or `entrypoints`, and adapters translate their driver's
exceptions into domain errors before they escape. If the project grew, an
import-linter contract in CI would be the cheap way to make the first one
automatic.

## Alternatives that were considered

### Web framework: FastAPI, kept Flask, or Litestar

Chose FastAPI. Flask 2.3 is what the original used and would have been a smaller
diff, but every request parameter would still be validated by hand — which is
how the threshold bug (finding 7) got in — and there would be no schema for a
client to generate against.

The trade-off is real: FastAPI's dependency injection and `Annotated` forms are
more machinery than a five-endpoint service strictly needs, and its async
routing invites the mistake of running a blocking forward pass on the event
loop, which this codebase avoids by pushing inference to a threadpool
explicitly. Litestar is a reasonable third option with a cleaner DI story. It
lost on familiarity: most teams already have FastAPI experience on hand. See [ADR 0002](adr/0002-fastapi-over-flask.md).

### Persistence: Postgres, MySQL, or keeping Mongo

Chose PostgreSQL, with MySQL and SQLite supported by the same adapter through
dialect-specific upserts. Counts are a tiny relational fact — one row per class
— with one operation that has to be atomic. `INSERT … ON CONFLICT DO UPDATE SET
count = count + excluded.count` does it in a single statement with no
application-level locking and no read-modify-write window.

Mongo's `$inc` is equally atomic per document, so the original was not wrong; it
was just a document store holding two columns. The assignment asked for
relational, and the schema being reviewable in a migration is worth more here
than schema flexibility nobody needs. See [ADR
0003](adr/0003-atomic-upsert.md).

Sharding or a counter service (Redis `INCRBY`, then flush) would be the next
step if writes ever outgrew one Postgres, but "outgrew" means tens of thousands
of images a second, and pretending otherwise now would be architecture theatre.

### Model serving: remote server, in-process, or both

Both, because they answer different constraints. TensorFlow Serving keeps the
model out of the API process, versions it independently and can be scaled on
GPU nodes while the API stays cheap; the price is a network hop and, over REST,
a JSON payload of pixel integers. In-process ONNX Runtime removes the hop and
the serialisation, at the cost of loading weights into every replica's memory
and coupling deploys of the API to deploys of the model.

Rather than choosing, the registry makes the choice a catalog entry. A model
declared `framework: tensorflow-serving` is called over HTTP; one declared
`framework: onnx` runs in-process. Callers see the same endpoint either way.
See [ADR 0004](adr/0004-model-registry.md) and
[MULTI_FRAMEWORK.md](MULTI_FRAMEWORK.md).

### Where thresholding happens

In the domain, not in the adapter. It is tempting to pass the threshold to the
model and let it filter, which TFS and most exported detectors accept, and it
would cut the response size. It also makes the caller's threshold
unverifiable, makes two backends behave differently at the same threshold, and
makes `/object-detect` and `/object-count` capable of disagreeing.

So detectors return everything they produced and `over_threshold` decides. The
one exception is the NMS score floor inside the ONNX post-processing (0.05 by
default), which exists to keep suppression cheap and is documented per adapter;
it sits well below any threshold a caller would send.

## Request path

`POST /object-count` with a 2 MB JPEG at threshold 0.9:

1. `MaxBodySizeMiddleware` compares `Content-Length` to the limit.
2. `RequestContextMiddleware` assigns a request id, starts the timer.
3. FastAPI parses the multipart form; a malformed one is a 422 before any
   handler runs.
4. The route reads the bytes, checks the real length, and builds a
   `domain.Image`. Bytes, not a file handle — the original passed a `BytesIO`
   that the detector consumed and the debug drawer then re-read, which works
   only because PIL happens to seek.
5. `services.count_action(model_name)` resolves the model through the registry
   (404 if unknown, 503 if its artifact will not load) and builds the use case.
6. `run_in_threadpool` runs it, so the event loop stays free.
7. `DetectObjects` validates the threshold, calls the detector, filters.
8. `count` groups by class; the repo increments atomically and reads the totals
   back.
9. The schema serialises; the middleware logs method, path, status and duration
   as one JSON line.

Every failure in that path has a defined status code and an error body with the
same shape.

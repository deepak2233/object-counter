# Code review of the original service

Assignment tasks 3 and 4. Nineteen findings against
[arenan02/object-counter](https://github.com/arenan02/object-counter) at
`master`, ordered by what would hurt first in production. All nineteen are
fixed in this rewrite. Four further changes were considered and rejected; the
last section says why.

Severity: **S1** loses or corrupts data, or takes the service down. **S2**
returns wrong answers or wrong status codes. **S3** slows people down.

| # | Sev | Finding | Status |
| --- | --- | --- | --- |
| 1 | S1 | Lost counts under concurrency | Fixed |
| 2 | S1 | Debug JPEGs written to disk on every request, from the domain layer | Fixed |
| 3 | S1 | No timeout on the TensorFlow Serving call | Fixed |
| 4 | S1 | `app.run(debug=True)` hardcoded | Fixed |
| 5 | S2 | Float class ids never match the int-keyed label map | Fixed |
| 6 | S2 | Any non-RGB image raises a 500 | Fixed |
| 7 | S2 | No threshold validation | Fixed |
| 8 | S2 | `model_name` accepted and ignored | Fixed |
| 9 | S2 | `read_values` returns `None` for unseen classes | Fixed |
| 10 | S2 | Environment variable drives an arbitrary `globals()` lookup | Fixed |
| 11 | S2 | Label map loaded from a CWD-relative path | Fixed |
| 12 | S2 | Detector errors surface as 500s with driver detail | Fixed |
| 13 | S2 | Whole image serialised as JSON integers | Fixed |
| 14 | S3 | A new MongoClient per repository call | Fixed (adapter replaced) |
| 15 | S3 | `print` instead of logging | Fixed |
| 16 | S3 | No health or readiness endpoints | Fixed |
| 17 | S3 | Tests assert almost nothing; no repo, integration or e2e coverage | Fixed |
| 18 | S3 | Setup is a page of shell commands, no lockfile, no CI | Fixed |
| 19 | S3 | Uploads unbounded in size | Fixed |

## The four that would page someone

### 1. Lost counts under concurrency (S1)

`CountInMemoryRepo.update_values` reads, adds in Python, writes back:

```python
stored_object_count = self.store[key]
self.store[key] = ObjectCount(key, stored_object_count.count + new_object_count.count)
```

Two threads counting one cat each can both read 4 and both write 5. Flask serves
requests on threads by default, so this is reachable with two concurrent
uploads. The Mongo adapter got this right with `$inc`; the in-memory one did
not, which means the dev and prod adapters disagreed about the one invariant the
product has.

Fixed twice over: `InMemoryObjectCountRepo` holds a lock, and
`SqlObjectCountRepo` issues `INSERT … ON CONFLICT DO UPDATE SET count = count +
excluded.count`, one statement, atomic in the database. There is a test that
fires 200 parallel increments at Postgres and asserts the total is 200
(`tests/integration/test_sql_repo.py`).

### 2. Debug images written on every request, from the domain (S1)

```python
def execute(self, image, threshold) -> CountResponse:
    predictions = self.__find_valid_predictions(image, threshold)  # writes two JPEGs
```

`__debug_image` is guarded by `if __debug__`, which is `True` in every normal
Python run, and `python -O` is not how anyone starts a Flask app. So each request
wrote `tmp/debug/all_predictions.jpg` and
`tmp/debug/valid_predictions_with_threshold_0.9.jpg`, with fixed filenames.
Concurrent requests overwrite each other's files, the disk fills up, and two
extra JPEG encodes sit in the latency path.

The deeper problem is architectural. `counter/domain/actions.py` imports PIL and
`counter.debug`, so the domain depends on an image library and on the local
filesystem. The dependency rule that the README describes is broken in the one
file that is supposed to show it off.

Fixed by deleting the behaviour from the use case. Drawing boxes on an image is
a client concern; the endpoint returns normalised coordinates and anything can
render them.

### 3. No timeout on the model call (S1)

```python
response = requests.post(self.url, data=predict_request)
```

No `timeout`, so the default is none. A TensorFlow Serving instance that accepts
the connection and then stops answering pins that worker forever. Enough of them
and the service is down while every health check still passes.

Fixed: `httpx.Client(timeout=30, transport=httpx.HTTPTransport(retries=2))`,
configurable through `COUNTER_TFS_TIMEOUT_SECONDS`, with connection errors
retried at the transport and everything else mapped to a 503.

### 4. `debug=True` in the entrypoint (S1)

```python
app.run('0.0.0.0', debug=True)
```

The Werkzeug debugger executes arbitrary Python from the browser on any
unhandled exception. The README tells you to run this module directly in
production (`ENV=prod python -m counter.entrypoints.webapp`). That is remote code
execution behind one 500.

Fixed: the server is uvicorn, reload and debug are off, and the profile is a
typed setting rather than a flag someone forgets.

## The wrong-answer class

### 5. Float class ids against an int-keyed label map (S2)

```python
self.classes_dict = {label['id']: label['display_name'] for label in labels}   # int keys
...
class_name = self.classes_dict[detection_class]                                # 17.0, a float
```

TensorFlow Serving returns `detection_classes` as floats over JSON. `d[17.0]`
happens to find `d[17]` in Python because `hash(17.0) == hash(17)`, so this
survives until a model returns `17.5` for a padded slot, or the ids arrive as
strings from a different serving version. Then every detection raises KeyError
inside the request. The fake detector never exercised this path, so no
test would have caught it.

Fixed in `class_name_for`, which coerces through `int()` and falls back to
`class_<id>` for ids missing from the map. Dropping unknown detections silently
would corrupt the counts, which are the product.

### 6. Any non-RGB image is a 500 (S2)

```python
np.array(image_.getdata()).reshape((im_height, im_width, 3))
```

A grayscale JPEG has one band, an RGBA PNG has four; both raise `ValueError:
cannot reshape array`. A CMYK TIFF has four. The service accepts any upload and
crashes on a large share of real ones. `getdata()` also builds a Python list of
tuples before numpy sees it, which is roughly an order of magnitude slower than
`np.asarray`.

Fixed in `adapters/detector/images.decode`: `convert("RGB")` first,
`np.asarray` after, `PIL.DecompressionBombError` caught, and everything
undecodable answered with 415 instead of 500. Tested against RGBA and grayscale
fixtures.

### 7. Threshold accepted unvalidated (S2)

```python
threshold = float(request.form.get('threshold', 0.5))
```

`threshold=high` raises ValueError inside the handler and returns a 500.
`threshold=90` is accepted, silently returns zero objects, and looks like a
model failure to the caller, since the confidence score is a probability and 90
can never match anything. Negative thresholds pass too.

Fixed: `validate_threshold` in the domain rejects anything outside `[0, 1]` and
anything non-numeric, mapped to 422 with the offending value in the message.

### 8. `model_name` accepted and ignored (S2)

```python
model_name = request.form.get('model_name', "rfcn")   # never used again
```

The parameter is read and dropped. A caller asking for a specific model gets
whatever the process was configured with and no indication that the request was
not honoured.

Fixed: `model_name` resolves through the registry, an unknown name is a 404 that
lists what is available, and the response says which model answered.

### 9. `read_values` returns `None` entries (S2)

```python
return [self.store.get(object_class) for object_class in object_classes]
```

Asking for a class that was never counted yields `[None]`, so every caller has
to filter. The Mongo adapter instead omitted the row entirely. Two
implementations of one port, two different contracts.

Fixed: the port documents zero, and both adapters return `ObjectCount(name, 0)`.

### 10. Environment variable to `globals()` lookup (S2)

```python
count_action_fn = f"{env}_count_action"
return globals()[count_action_fn]()
```

`ENV=prd` fails with `KeyError: 'prd_count_action'` at the first request rather
than at startup. Any module-level name becomes reachable from an environment
variable. There is no type checking and no way to enumerate valid profiles.

Fixed: `Settings` is a pydantic model with `Literal["dev", "test", "prod"]`, and
`bootstrap.py` branches explicitly. `COUNTER_ENV=prd` now fails at boot with a
message naming the field and the allowed values.

### 11. Label map path relative to the working directory (S2)

```python
with open('counter/adapters/mscoco_label_map.json') as json_file:
```

Works from the repo root, fails from anywhere else — a container with a
different WORKDIR, a systemd unit, a test run from a subdirectory, or the
installed package. The file is package data and was not declared as such.

Fixed: `importlib.resources` for packaged maps, and an absolute path for maps
that belong to a particular model.

### 12. Adapter failures leak as 500s (S2)

`response.json()['predictions'][0]` raises KeyError when TFS answers 503
"model not loaded" or when a proxy returns HTML. The caller sees a 500 and a
stack trace shape they can do nothing with.

Fixed: adapters translate to `DetectorUnavailableError` and `RepositoryError`;
one exception handler maps those to 503 and logs the detail server-side. Nothing
from httpx, SQLAlchemy or onnxruntime reaches a response body.

### 13. The image as a JSON array of integers (S2/S3)

```python
predict_request = '{"instances" : %s}' % np.expand_dims(np_image, 0).tolist()
```

A 4000×3000 photo is 36 million integers, roughly 90 MB of JSON, built as one
Python string. Encoding is CPU-bound in the request path, and TFS parses the
same 90 MB back.

Fixed: images are downscaled so the longest side is at most
`COUNTER_MAX_IMAGE_SIDE` (1024 by default) before serialisation, which costs a
little accuracy on small objects and bounds both latency and memory. Sending
JPEG bytes to a model with a `tf.string` input signature would be better again,
but it needs a re-export of the SavedModel, which is out of scope here.

## The rest

**14 — a MongoClient per call.** `__get_counter_col` builds a client on every
read and every write. `MongoClient` owns a connection pool and is meant to live
for the process. The Mongo adapter is gone here: the assignment asked for a
relational one, and keeping two persistence backends alive for one exercise
means two things to test and two to keep honest. `ObjectCountRepo` is unchanged,
so reinstating it is one file.

**15 — `print` instead of logging.** `print(predictions)` dumps model output
into stdout on every request. No level, no request id, no sampling, unparseable
by anything downstream. Replaced with structured JSON logs and a request id that
appears in the access line, in every domain log, in the error body and in the
`X-Request-ID` header.

**16 — no health endpoints.** Nothing for an orchestrator to probe. Added
`/healthz` (liveness, checks nothing else on purpose: a liveness probe that
depends on the database restarts healthy pods during a database incident) and
`/readyz`, which checks that the store is reachable *and* migrated, and that the default model
loads.

**17 — tests that pass regardless.** `assert json.loads(response.data) != None`
is true for `{}`, `[]` and `0`. The suite covered two pure functions and one
endpoint against the fake detector; no repository test, no error path, nothing
that would catch findings 1, 5, 6, 7 or 9. Replaced with 165 tests across three
levels; see [TESTING.md](TESTING.md).

**18 — setup by copy-paste.** Two shell blocks for TensorFlow Serving (one Unix,
one PowerShell), a `wget`, four `mv`s, and a `docker run` per service. No
lockfile, dev and runtime dependencies mixed in one `requirements.txt`, and
nothing runs in CI. Replaced with a Makefile where each task is one target,
Docker Compose for the backing services, dependency groups in `pyproject.toml`,
and a GitHub Actions workflow that runs lint, types and tests against a real
Postgres.

**19 — unbounded uploads.** `request.files['file']` with no size limit; a
multi-gigabyte POST is buffered before anything checks it. Now
`Content-Length` is rejected above the limit at the middleware, and the received
bytes are checked again in the route, because chunked uploads do not declare one.

## Left alone deliberately

**gRPC to TensorFlow Serving.** Faster than REST and avoids the JSON encode, but
it needs the `tensorflow-serving-api` package, which pulls TensorFlow. That is a
2 GB dependency in an image whose Python side is 200 MB, and the downscale in
finding 13 already removes most of the cost.

**Async all the way down.** Inference and psycopg are blocking, and the routes
push them to a threadpool. Going fully async would mean asyncpg and an async
inference client, and would gain nothing while a forward pass holds the CPU.

**Idempotency keys on `/object-count`.** A retried upload counts twice. Fixing
it properly needs a client-supplied key and a dedupe table; the assignment has
no requirement for it, and inventing one would be scope creep. Worth raising
with whoever owns the product.

**Authentication.** There is none, and there was none. In a real deployment this
sits behind a gateway that terminates auth; inventing a scheme here would be
worse than nothing.

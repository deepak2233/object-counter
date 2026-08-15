# 2. FastAPI instead of Flask

Date: 2026-08-13
Status: accepted

## Context

The original entrypoint is Flask. Request parsing is manual:

```python
threshold = float(request.form.get('threshold', 0.5))
uploaded_file = request.files['file']
model_name = request.form.get('model_name', "rfcn")
```

`threshold=high` is a 500. `threshold=90` is accepted and silently matches
nothing. A missing file raises a werkzeug `BadRequestKeyError`. `model_name` is
read and never used. There is no schema, so a client has nothing to generate
against, and the response shape is whatever `jsonify` makes of a dataclass.

## Decision

FastAPI, with pydantic schemas for request and response, an exception handler
per domain error type, and blocking work pushed to a threadpool with
`run_in_threadpool`.

## Consequences

Validation happens before any handler runs, and the OpenAPI document at
`/openapi.json` is generated from the same types the code uses, so it cannot
drift. Error bodies share one envelope with a request id in it. The threadpool
detail matters: inference and psycopg both block, and running them on the event
loop would stall every other in-flight request for the length of a forward pass.

Costs, stated plainly. It is a bigger dependency than Flask and brings starlette
and pydantic v2 with it. Its dependency injection and `Annotated` form
parameters are more machinery than this service needs. And its async routing
invites exactly the mistake described above, which this codebase avoids by being
explicit about it.

Litestar was the other candidate, with a cleaner DI story. Rejected on
familiarity: FastAPI is the framework a team is most likely to already know, and
for a service this size that matters more than the design of its container.

The `/object-count` request and response contract is unchanged, so existing
clients of the original service keep working.

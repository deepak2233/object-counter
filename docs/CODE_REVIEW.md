# Code review

This review covers the original `arenan02/object-counter` implementation used
for the assignment. Severity is based on service impact:

- S1: availability, security, or data integrity
- S2: incorrect behavior or API contract
- S3: maintainability or delivery risk

## Findings

| # | Sev | Finding | Change in this repository |
| --- | --- | --- | --- |
| 1 | S1 | In-memory updates use read-modify-write and lose concurrent increments | Lock in memory; atomic database upsert in SQL |
| 2 | S1 | Detection writes fixed-name debug images on every normal request | Removed file output from the use case |
| 3 | S1 | TensorFlow Serving request has no timeout | Bounded client timeout and transport retries |
| 4 | S1 | The server runs with debug mode enabled | Uvicorn entrypoint with debug disabled |
| 5 | S2 | Serving output and class IDs are not normalized or validated | Dedicated parser, integer normalization, fallback labels |
| 6 | S2 | Decoder assumes three-channel RGB input | Decode once, apply EXIF orientation, convert to RGB |
| 7 | S2 | Threshold accepts invalid strings and values outside 0 to 1 | Domain validation mapped to HTTP 422 |
| 8 | S2 | `model_name` is accepted but ignored | Catalog-backed model registry and 404 for unknown names |
| 9 | S2 | Repository implementations disagree on unseen classes | Both return an explicit zero count |
| 10 | S2 | Environment input selects an arbitrary `globals()` entry | Typed settings and explicit composition |
| 11 | S2 | Label map depends on the current working directory | Package-aware and explicit path loading |
| 12 | S2 | HTTP, model, and database exceptions escape as 500 responses | Adapter errors mapped to domain errors and safe API messages |
| 13 | S2 | Full-resolution images become large JSON integer arrays | Configurable longest-side downscale before TFS calls |
| 14 | S3 | A Mongo client is created for each repository operation | Long-lived SQLAlchemy engine and connection pool |
| 15 | S3 | Model output is printed directly | Structured logging with request IDs |
| 16 | S3 | No liveness or readiness endpoints | Separate `/healthz` and dependency-aware `/readyz` |
| 17 | S3 | Tests do not cover repository or failure behavior | Unit, integration, and HTTP-level suites |
| 18 | S3 | Setup has no lock or automated verification | `uv.lock`, Make targets, and CI |
| 19 | S3 | Upload size is unbounded | Header guard, streamed byte limit, and decoded pixel limit |

## Main design corrections

### Atomic counts

The SQL repository increments with a single dialect-specific upsert. The key is
`(model_name, model_version, object_class)`, so totals from different model
versions cannot be mixed. Migration `0002` keeps old rows under
`legacy/unknown`.

### Model calls

TensorFlow Serving uses a persistent client with a timeout. Readiness calls its
model-status endpoint and requires an available version. Catalog names may map
to a different TFS `remote_name` without changing the name returned to API
clients.

In-process models load lazily through the same registry. Model resolution and
inference both run outside the event loop. The registry closes owned clients
and model resources during application shutdown.

### Image handling

The route reads uploads in bounded chunks. The shared decoder limits decoded
pixels, applies EXIF orientation, and converts supported formats to RGB.
YOLO-style output uses class-aware non-maximum suppression so overlapping
objects from different classes are retained.

### Public errors

Backend details are logged but not returned. Early middleware failures use the
same error envelope and request ID as route failures. Caller-supplied request IDs
are accepted only when they match a short, restricted format.

### Model artifacts

S3 artifacts require a SHA-256 digest. Downloads use unique staging files and an
atomic rename after checksum verification. Optional backend dependencies are
selected at install or image-build time.

## Deliberate limits

- TFS still receives JSON pixels. A byte-input SavedModel or gRPC client would
  require a different serving contract.
- `/object-count` has no idempotency key, so a client retry counts again.
- Authentication is expected at a gateway; the service does not implement an
  identity scheme.
- Pull-request CI tests adapter contracts with stubs. Each deployment still
  needs a test using its real model artifact and runtime wheel.

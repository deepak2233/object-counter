# 4. A model catalog and registry, not a hardcoded detector

Date: 2026-08-13
Status: accepted

## Context

Two of the assignment questions turn out to be the same question. Task 5 asks
what changes to serve several internally trained models; task 6b asks what
changes to serve several deep learning frameworks. Both are "the service must
resolve a model by name at request time, and how it runs is a property of that
model".

The original hardcodes both: `TFSObjectDetector(tfs_host, tfs_port, 'rfcn')` in
`config.py`, with the label map read from a path relative to the working
directory.

## Decision

A third port, `ObjectDetectorRegistry`, with `get(name) -> ObjectDetector` and
`available() -> list[ModelInfo]`. Its implementation reads a YAML catalog
validated by pydantic, and builds the detector the entry declares:
`tensorflow-serving`, `onnx`, `torchscript` or `fake`.

Detectors are built on first use and cached under a lock. Artifacts are resolved
by an `ArtifactStore` that supports local paths and `s3://`, and verifies a
SHA-256 when the entry pins one.

## Consequences

Adding a model is a catalog entry plus an artifact. Adding a framework is one
adapter file and one branch. Neither touches the domain, the routes or the other
adapters.

`model_name` on a request now does something, an unknown name is a 404 listing
the alternatives, and `GET /models` says what an instance can serve and what it
has loaded.

Lazy loading means the service starts without every model present, and a model
that fails to load only fails the requests that ask for it —
`COUNTER_PRELOAD_MODELS=true` trades boot time for first-request latency where
that is preferred.

The costs: an indirection between a request and its detector, a YAML file that
is now part of the deployment contract, and a per-request dictionary lookup
(measured in nanoseconds, ignore it). The catalog is a file rather than a model
registry, which is the right size for this service and the wrong size for fifty
models — `MULTI_MODEL.md` names that boundary and where MLflow or Vertex would
plug in.

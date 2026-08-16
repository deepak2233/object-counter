# 4. Use a model catalog and registry

Date: 2026-08-13
Status: accepted

## Context

A hardcoded TensorFlow Serving model cannot support request-time model selection,
private artifacts, version reporting, or multiple runtimes.

## Decision

Add an `ObjectDetectorRegistry` that resolves a catalog name to an
`ObjectDetector`. The YAML or JSON entry defines framework, version, labels,
endpoint or artifact, and preprocessing settings.

Detectors load on first use, are cached under a lock, and close during
application shutdown. S3 artifacts require a SHA-256 digest.

## Consequences

Adding a model changes deployment configuration, not domain or route code.
Adding a framework requires an adapter, catalog type, optional dependency group,
and registry construction branch.

Lazy loading reduces startup cost but moves model load latency to the first
request. `COUNTER_PRELOAD_MODELS` can move that work to startup. Real model
artifacts still require release-time contract tests.

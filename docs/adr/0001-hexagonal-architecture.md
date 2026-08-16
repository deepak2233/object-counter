# 1. Keep ports and adapters

Date: 2026-08-13
Status: accepted

## Context

The original use case imported Pillow and wrote debug images. This coupled
domain behavior to image decoding and the local filesystem.

## Decision

Keep domain models and use cases independent of infrastructure.

- Images cross the detector port as bytes and metadata.
- Detectors return normalized domain predictions.
- Image decoding stays in detector adapters.
- Debug image output is removed.

`counter/bootstrap.py` constructs the selected adapters.

## Consequences

Use cases can be tested without web, database, or inference dependencies.
FastAPI, the CLI, SQL, and each detector can change independently.

The service has additional interfaces and a composition root. That indirection
is justified because persistence and detector backends are explicit extension
points.

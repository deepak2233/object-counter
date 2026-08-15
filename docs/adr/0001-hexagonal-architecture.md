# 1. Keep hexagonal architecture, and enforce the dependency rule

Date: 2026-08-13
Status: accepted

## Context

The original service is organised as domain, adapters and entrypoints, and its
README explains the layering. The code does not hold the line:
`counter/domain/actions.py` imports PIL and `counter.debug`, and writes JPEGs to
`tmp/debug/` inside the use case. So the layer that is meant to be free of
infrastructure depends on an image library and on the local filesystem.

## Decision

Keep the architecture. Enforce the rule that `counter.domain` imports nothing
from `counter.adapters` or `counter.entrypoints`, and move image handling into
the adapters.

Concretely: images cross the boundary as `domain.Image` (bytes plus filename and
content type), detectors return `Prediction` objects with normalised boxes, and
the debug drawing is deleted rather than relocated — rendering boxes is a client
concern.

## Consequences

The domain is testable with no dependencies beyond pytest, and the 119 unit
tests run in 1.3 seconds. The end-to-end tests can drive the real HTTP stack
against a fake detector because `create_app(services=...)` takes its
collaborators as arguments.

The cost is indirection that a five-endpoint service does not strictly need:
three ports, a registry, a composition root. That price buys the two things the
assignment asks for later — swapping the persistence adapter and serving several
frameworks — without touching a use case.

Bytes rather than a file handle is a small decision with a real payoff: a
`BinaryIO` is stateful, and the original relied on PIL seeking back to zero after
the detector had consumed the stream.

The rule is currently enforced by review. If the project grew, an import-linter
contract in CI would make it mechanical.

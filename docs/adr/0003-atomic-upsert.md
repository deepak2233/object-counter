# 3. Use PostgreSQL atomic upserts

Date: 2026-08-13
Status: accepted

## Context

Counts must accumulate under concurrent requests and must not mix results from
different model versions. A read-modify-write in application code can lose
increments.

## Decision

Store one row per model, version, and class:

```text
model_name
model_version
object_class
count
updated_at
```

The first three columns form the primary key. Writes use the dialect's atomic
incrementing upsert. Duplicate classes in one batch are merged before the
statement is built.

PostgreSQL is the deployment target. SQLite and MySQL use equivalent syntax in
the same adapter.

## Consequences

Concurrent database writers do not need an application lock or a separate read.
Model upgrades have independent totals.

Migration `0002` moves old rows to the `legacy/unknown` model scope. Tests
cover migration preservation, both migration directions, and concurrent
PostgreSQL increments.

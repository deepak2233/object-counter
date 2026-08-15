# 3. PostgreSQL with an atomic upsert for the counts

Date: 2026-08-13
Status: accepted

## Context

Assignment task 2 asks for a relational adapter for `ObjectCountRepo`. The port
has two operations: read the totals, and add to them.

The addition is the whole problem. The original in-memory adapter did it in
Python — read the row, add, write back — which loses counts when two requests
overlap. The Mongo adapter used `$inc` and did not.

## Decision

One table, `object_counts(object_class PK, count, updated_at)`, and one
statement per write:

```sql
INSERT INTO object_counts (object_class, count)
VALUES (:object_class, :count)
ON CONFLICT (object_class)
DO UPDATE SET count = object_counts.count + excluded.count, updated_at = now()
```

The class name is the primary key rather than a surrogate id, because the
uniqueness constraint is what makes the upsert atomic; a surrogate would need a
separate unique index to get the same guarantee back.

SQLite and MySQL get the same semantics through their own syntax
(`ON CONFLICT` and `ON DUPLICATE KEY UPDATE`), dispatched on the dialect name.
The batch is deduplicated in Python first, because a multi-row upsert naming the
same key twice is a runtime error on PostgreSQL.

## Consequences

Concurrent writers cannot lose counts, and there is no application-level lock,
no `SELECT … FOR UPDATE`, and no retry loop. A test fires 200 parallel
increments at Postgres and asserts the total is 200.

MySQL support is free and SQLite support means the whole test suite runs on a
laptop with nothing installed — while CI runs against Postgres, because SQLite
proves nothing about the PostgreSQL upsert or about concurrent writers.

A dialect with no branch raises `RepositoryError` naming it, rather than falling
back to something that looks like it works.

The schema ships as an Alembic migration, applied by a one-shot container in the
compose stack and by `make migrate` locally. Both directions are tested.

The limit of this design is a single row per class as a write hotspot. At tens
of thousands of images a second, the fix is a counter service (Redis `INCRBY`
flushed periodically) or per-shard rows summed on read. Neither is worth
building for a service whose current load is one reviewer with curl.

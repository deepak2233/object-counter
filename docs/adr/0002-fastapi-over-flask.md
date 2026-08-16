# 2. Use FastAPI

Date: 2026-08-13
Status: accepted

## Context

The original Flask route parsed multipart fields manually. Invalid thresholds
and missing files could become 500 responses, and there was no generated API
schema.

## Decision

Use FastAPI with Pydantic response schemas and centralized exception handlers.
Run blocking model and database work through Starlette's thread pool.

## Consequences

The API publishes OpenAPI, validates form input consistently, and returns one
error envelope. The existing `/object-count` field names remain available.

FastAPI adds Pydantic and Starlette and requires care around blocking calls. The
routes remain async only at the transport layer; inference and SQL are still
synchronous.

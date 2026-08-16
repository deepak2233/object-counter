# Internally trained models

Private models need a catalog, controlled artifact access, versioned labels, and
release checks. The service implements the catalog and artifact-loading parts.

## Catalog

```yaml
default_model: retail-shelf-v3

models:
  - name: retail-shelf-v3
    framework: onnx
    version: "3.2.1"
    labels: /app/config/labels/retail_shelf_v3.txt
    input_size: 640
    score_floor: 0.05
    iou_threshold: 0.45
    artifact:
      uri: s3://${MODEL_BUCKET}/retail-shelf/3.2.1/model.onnx
      sha256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
```

Set `COUNTER_MODEL_CATALOG` to the mounted YAML or JSON file. Environment
references are expanded when it loads. Catalog validation fails at startup for
duplicate names, invalid frameworks, missing artifacts, and unpinned S3
artifacts.

Adding a model does not require application code changes. It does require a
deployment configuration change and, when a new framework group is needed, a
runtime image containing that group.

## Artifact handling

`ArtifactStore` supports:

- a local path or `file://` URI for a mounted volume
- `s3://` for AWS S3 or a compatible service

S3 objects must be pinned by SHA-256. The downloader writes to a unique staging
file, verifies the digest, then atomically moves it into a digest-addressed
cache. Failed or partial downloads are not served.

Use workload identity, an instance role, or another platform-native credential
source. Do not place cloud keys or model files in the repository.

## Version behavior

The catalog version is returned by both detection endpoints. Accumulated counts
are keyed by model name and version. Deploying version 4 therefore starts a new
set of totals instead of mixing them with version 3.

Artifacts should be immutable. Publish a new path and digest for a retrained
model; do not overwrite an existing version.

## Deployment choices

For a few small models, one catalog and lazy loading can be enough. For large
models or GPU serving, prefer one model or model family per deployment and route
by name outside the service. This gives each model an independent memory budget,
autoscaling policy, and rollback.

The catalog file is suitable for this exercise. At larger scale it should be
generated from the team's model registry, such as MLflow, Vertex AI, or
SageMaker.

## Release checks still required

The repository cannot define these without the team's artifacts and data:

- real artifact load and inference
- label-map compatibility
- accuracy and drift gates
- model-specific latency and memory limits
- shadow or canary evaluation
- per-model operational metrics

`config/models.example.yaml` shows ONNX, TorchScript, and TensorFlow Serving
entries.

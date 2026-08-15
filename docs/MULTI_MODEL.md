# Serving internally trained models

Assignment task 5: *if we want to use multiple models trained internally (not
public), what would you change in the setup of the project?*

The original setup has one model, hardcoded. `TFSObjectDetector(tfs_host,
tfs_port, 'rfcn')` names it in `config.py`, the label map is a file inside the
adapter package, and the README tells you to `wget` the weights from a public
Google Storage bucket. None of that works for a model your team trained: there
is no public URL, the label set is not COCO, the artifact needs access control
and an audit trail, and there will be more than one of them.

Four things change, and three of them are already implemented here.

## 1. A catalog instead of a constant

Models are declared in a YAML file, validated at startup by pydantic. Adding one
is an entry plus an artifact in the model store — no code change, no rebuild,
no redeploy of the image.

```yaml
default_model: retail-shelf-v3

models:
  - name: retail-shelf-v3
    framework: onnx
    version: "3.2.1"
    labels: /app/config/labels/retail_shelf_v3.txt
    input_size: 640
    artifact:
      uri: s3://${MODEL_BUCKET}/detection/retail-shelf/3.2.1/model.onnx
      sha256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
```

Point `COUNTER_MODEL_CATALOG` at it. `config/models.example.yaml` is a filled-in
example with four models across three frameworks. `${VAR}` references are
expanded when the file is read, so one catalog is deployable to dev, staging and
production with different buckets and endpoints.

Callers select a model per request with `model_name`, and `GET /models` lists
what an instance can serve. An unknown name is a 404 that names the alternatives
rather than silently answering with the default.

Implemented in `counter/adapters/detector/catalog.py` and `registry.py`.

## 2. An artifact store, pinned by digest

`ArtifactStore.resolve()` turns a URI into a local path:

- `file://` or a bare path for a mounted volume — a PVC, an NFS mount, or a
  model baked into a sidecar image;
- `s3://bucket/key` for any S3-compatible store (AWS, MinIO, Ceph), using boto3
  credentials from the environment or the pod's role.

Downloads land in a `.part` file and are renamed only after the SHA-256 matches,
so a crashed download is never picked up as a valid cache entry by the next
process to start. The cache is digest-addressed under
`COUNTER_ARTIFACT_CACHE_DIR`, which in the compose stack is a named volume, so a
restart does not re-pull gigabytes.

If a digest is set and does not match, the model refuses to load. That is the
difference between "we serve retail-shelf 3.2.1" and "we serve whatever was in
the bucket this morning". If a digest is absent the service logs a warning and
proceeds, which is a deliberate escape hatch for local experiments.

A new scheme — GCS, Azure Blob, an MLflow or Vertex registry — is one
`_fetch_*` method in `artifacts.py`. The registry and the detectors do not
change.

Implemented in `counter/adapters/detector/artifacts.py`.

## 3. Credentials and access, which is mostly not code

- The bucket is private and the service reads it with a role, not a key checked
  into an env file. On EKS that is IRSA, on GKE workload identity, on a VM an
  instance profile. `COUNTER_*` settings carry no secrets; the only credential
  in the compose file is the local Postgres password.
- Model artifacts are immutable at a version. Retraining publishes 3.2.2; it
  never overwrites 3.2.1. Rollback is then a catalog change, not a rebuild.
- The catalog and the label files are configuration, mounted as a ConfigMap or
  baked into the deployment repo, so a model change is reviewable in a PR with a
  diff someone can read.
- Weights never enter git. `.gitignore` covers `*.onnx`, `*.pt`, `*.pb`, `tmp/`
  and `var/`, and the Docker build ignores them too.

## 4. What still needs building

Not implemented here, because they need decisions that belong to the team that
owns the models:

**A registry as the source of truth.** The catalog is a file. At more than a
handful of models you want MLflow, Vertex Model Registry or SageMaker as the
system of record, with the catalog generated from it at deploy time. The
`ArtifactStore` seam is where that plugs in.

**Warm-up and memory budget.** `COUNTER_PRELOAD_MODELS=true` loads every catalog
entry at boot; the default is lazy. With five in-process models on one replica,
memory is the constraint and the honest answer is usually one model per
deployment, routed by name at the gateway, rather than five in one process. The
registry supports both; the deployment decides.

**Per-model metrics.** Latency, error rate and detection counts should carry the
model name and version as labels, so a regression after a model bump is visible
without correlating deploys by hand. The logs already carry `model`; a
Prometheus exporter would be the next step.

**Shadow evaluation.** Running a candidate model alongside the incumbent on a
sample of live traffic, comparing counts, and promoting on evidence. The
registry makes it mechanically possible — resolve two detectors, call both — but
where the comparison is stored and who decides on promotion is a product
question.

## Trying it locally

```bash
mkdir -p var/local-models
# export your model, then:
cat > config/models.local.yaml <<'YAML'
default_model: my-detector
models:
  - name: my-detector
    framework: onnx
    version: "0.1.0"
    labels: var/local-models/classes.txt
    artifact:
      uri: var/local-models/model.onnx
YAML

COUNTER_MODEL_CATALOG=config/models.local.yaml \
COUNTER_ENV=prod make run-prod

curl localhost:5000/models
curl -F "file=@resources/images/cat.jpg" -F "threshold=0.6" \
     -F "model_name=my-detector" localhost:5000/object-detect
```

The ONNX extra is needed for that: `pip install -e ".[onnx]"`.

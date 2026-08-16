# Framework adapters

All backends implement `ObjectDetector` and return the same domain
`Prediction` objects.

| Framework | Execution | Install group | Supported contract |
| --- | --- | --- | --- |
| TensorFlow Serving | Remote REST | base | Detection SavedModel response |
| ONNX Runtime | In process | `onnx` | YOLOv8-style tensor |
| TorchScript | In process | `torch` | YOLOv8 tensor or TorchVision detection dict |
| Fake | None | base | Development and tests only |

Install a runtime locally with uv:

```bash
uv sync --frozen --extra dev --extra onnx
uv sync --frozen --extra dev --extra torch
```

For an image build, pass the needed groups:

```bash
docker build \
  --build-arg 'UV_SYNC_ARGS=--extra onnx --extra s3' \
  -t object-counter .
```

## Shared contract

The domain sees image bytes in and normalized predictions out. Tensor layout,
device placement, label lookup, non-maximum suppression, and serving protocol
stay inside the adapter.

ONNX preprocessing uses letterbox resize, NCHW float32 input, and class-aware
NMS. TorchScript has an explicit `input_contract`:

- `yolov8`: batched NCHW tensor with YOLO-style output
- `torchvision`: list of CHW tensors with `boxes`, `labels`, and `scores`

The contract must match the export. Detecting it from output shape after a bad
forward pass is too late.

## Test scope

Pull-request tests inject an ONNX session or Torch module. They verify input
shape, output conversion, box normalization, NMS, and error translation without
installing large framework wheels.

This does not prove that a specific model file loads or that its preprocessing
matches training. Release verification for each model should run with:

- the real runtime wheel
- the exact pinned artifact
- representative images
- expected labels, boxes, and latency limits

## Adding another backend

1. Implement `ObjectDetector` in `counter/adapters/detector/`.
2. Add the framework literal and catalog fields.
3. Add one registry construction branch.
4. Put its dependency in an optional group.
5. Test the adapter contract with an injected runtime and one real artifact in
   deployment verification.

No detection or counting use-case change is required.

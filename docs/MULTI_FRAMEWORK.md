# Several deep learning frameworks

Assignment task 6b. Three backends are implemented behind one port, and adding a
fourth is one file plus one branch.

| Framework | Adapter | Where inference runs | Install |
| --- | --- | --- | --- |
| TensorFlow Serving | `tensorflow_serving.py` | Remote, over REST | included |
| ONNX Runtime | `onnx_runtime.py` | In-process | `pip install -e ".[onnx]"` |
| TorchScript | `torchscript.py` | In-process | `pip install -e ".[torch]"` |
| — | `fake.py` | Nowhere | included |

## What makes it work

The port is four lines:

```python
class ObjectDetector(ABC):
    @property
    def info(self) -> ModelInfo: ...
    def predict(self, image: Image) -> list[Prediction]: ...
```

`Image` is bytes, `Prediction` is a class name, a score and a box normalised to
`0..1`. Every framework-specific concept — session options, execution providers,
tensor layout, letterbox padding, NMS, label maps — stays inside its adapter. The
use cases have no idea which one answered.

Two consequences worth stating. The domain never sees a tensor, so the counting
logic is identical across backends by construction. And because boxes are
normalised, a client rendering them does not need to know what input resolution
the model ran at.

## The three backends

**TensorFlow Serving** posts the pixel array as JSON to
`/v1/models/<name>:predict`. Details in [CODE_REVIEW.md](CODE_REVIEW.md#13--the-image-as-a-json-array-of-integers-s2s3):
bounded timeout, transport retries, downscaling before serialisation, and a
parser that tolerates truncated arrays and float class ids.

**ONNX Runtime** loads a session once, then per request: letterbox to a square,
convert to NCHW float32 in `0..1`, run, decode. The decode handles the YOLOv8
export layout `(1, 4 + num_classes, num_boxes)` and its transpose, picks the
argmax class, runs greedy NMS, and maps boxes back through the letterbox
transform to normalised coordinates.

The pre- and post-processing lives in `onnx_ops.py`, which imports numpy and
nothing else. That is deliberate: letterbox padding, xywh/xyxy confusion, NMS
and coordinate mapping are where detector bugs actually live, and this way they
are tested in milliseconds without a 200 MB wheel installed. `tests/unit/test_onnx_ops.py`
covers the letterbox arithmetic, the transposed layout, suppression of duplicate
boxes, and clipping of boxes that extend past the image.

**TorchScript** mirrors the ONNX path and reads both output conventions: the
torchvision detection dict (`boxes`, `labels`, `scores` in input-pixel space)
and a raw tensor, which goes through the same decoder as ONNX.

## Testing without the wheels

Both in-process adapters take an injected session or module:

```python
detector = OnnxObjectDetector(labels=LABELS, session=StubSession(fake_output))
```

`InferenceSession` is a `Protocol` covering the two methods the adapter calls, so
the stub is nine lines and the tests assert on what actually matters: that the
tensor fed to the model has shape `(1, 3, 640, 640)`, dtype float32 and values in
`0..1`; that a `RuntimeError` from the session becomes a 503-mapped
`DetectorUnavailableError` rather than a 500; that an unreadable output shape is
reported as such.

CI therefore proves the adapters are correct without installing onnxruntime or
torch. What it does not prove is that a particular `.onnx` file loads — that
needs the real wheel and a real artifact, and belongs in a nightly job with the
model store attached, not in the pull-request loop.

## Adding a fourth

For OpenVINO, TensorRT, a Triton client or a plain TensorFlow SavedModel:

1. Write `counter/adapters/detector/<framework>.py` with a class implementing
   `ObjectDetector`. Reuse `images.decode`, and `onnx_ops` if the output is
   YOLO-shaped.
2. Add the literal to `Framework` in `catalog.py` and a branch in
   `CatalogModelRegistry._build`.
3. Add the dependency as an optional extra in `pyproject.toml`, imported lazily
   inside the adapter so the other deployments do not carry it.
4. Test it with an injected session, the way the ONNX adapter is tested.

Nothing in `domain/`, `entrypoints/` or the other adapters changes. That is the
whole claim the architecture is making, and it is checkable: `grep -r "onnx\|torch\|tensorflow" counter/domain counter/entrypoints`
returns nothing.

## What the frameworks cost

Remote serving (TFS, Triton) versus in-process (ONNX, TorchScript) is the real
decision, and it is not about frameworks at all:

- Remote keeps weights out of the API process, so the API scales on CPU while
  models scale on GPU, and a model update is independent of an API deploy. It
  costs a network hop, and over the TFS REST API, a JSON encode of every pixel.
- In-process removes both costs and the operational surface of a second
  service. It puts the weights in every replica's memory and ties model updates
  to API deploys.

Below a few requests a second, in-process ONNX on CPU is simpler and usually
faster end to end. Above that, or with a GPU in play, remote serving wins on
utilisation. The registry means that decision can be revisited per model without
touching the service.

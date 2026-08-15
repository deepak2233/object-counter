"""Image decoding shared by every detector adapter.

PIL lives here and only here on the inference side, so the domain and the ports
stay free of it.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image as PilImage
from PIL import UnidentifiedImageError

from counter.domain.errors import InvalidImageError
from counter.domain.models import Image


def decode(image: Image, max_bytes: int | None = None) -> PilImage.Image:
    """Decode bytes into an RGB image.

    Always converts to RGB. The upstream implementation reshaped the raw pixel
    buffer to `(height, width, 3)`, which raises on any grayscale, CMYK or RGBA
    input — a PNG with an alpha channel was enough to return a 500.
    """
    if not image.content:
        raise InvalidImageError("image payload is empty")

    if max_bytes is not None and len(image.content) > max_bytes:
        raise InvalidImageError(f"image is {len(image.content)} bytes, limit is {max_bytes} bytes")

    try:
        with PilImage.open(BytesIO(image.content)) as decoded:
            decoded.load()
            return decoded.convert("RGB")
    except PilImage.DecompressionBombError as exc:
        # A 50k x 50k PNG is a few hundred KB on the wire and gigabytes in RAM.
        raise InvalidImageError("image is too large to decode safely") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"payload is not a decodable image: {exc}") from exc


def resize_longest_side(pil_image: PilImage.Image, max_side: int) -> PilImage.Image:
    """Downscale so the longest side is at most `max_side`, preserving aspect.

    Boxes are normalised, so downscaling costs a little accuracy on small objects
    and buys a bounded request payload and a bounded latency.
    """
    longest = max(pil_image.size)
    if max_side <= 0 or longest <= max_side:
        return pil_image

    scale = max_side / longest
    new_size = (max(1, round(pil_image.width * scale)), max(1, round(pil_image.height * scale)))
    return pil_image.resize(new_size, PilImage.Resampling.BILINEAR)


def to_uint8_array(pil_image: PilImage.Image) -> np.ndarray:
    """(height, width, 3) uint8 array. `np.asarray` avoids a copy where it can."""
    return np.asarray(pil_image, dtype=np.uint8)

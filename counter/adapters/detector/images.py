"""Image decoding shared by detector adapters."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image as PilImage
from PIL import ImageOps, UnidentifiedImageError

from counter.domain.errors import InvalidImageError
from counter.domain.models import Image

MAX_IMAGE_PIXELS = 25_000_000


def decode(
    image: Image,
    max_bytes: int | None = None,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> PilImage.Image:
    """Decode bytes into an oriented RGB image."""
    if not image.content:
        raise InvalidImageError("image payload is empty")

    if max_bytes is not None and len(image.content) > max_bytes:
        raise InvalidImageError(f"image is {len(image.content)} bytes, limit is {max_bytes} bytes")

    try:
        with PilImage.open(BytesIO(image.content)) as decoded:
            if decoded.width * decoded.height > max_pixels:
                raise InvalidImageError(f"image exceeds the {max_pixels} pixel limit")
            decoded.load()
            return ImageOps.exif_transpose(decoded).convert("RGB")
    except PilImage.DecompressionBombError as exc:
        # A 50k x 50k PNG is a few hundred KB on the wire and gigabytes in RAM.
        raise InvalidImageError("image is too large to decode safely") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"payload is not a decodable image: {exc}") from exc


def resize_longest_side(pil_image: PilImage.Image, max_side: int) -> PilImage.Image:
    """Downscale while preserving the aspect ratio."""
    longest = max(pil_image.size)
    if max_side <= 0 or longest <= max_side:
        return pil_image

    scale = max_side / longest
    new_size = (max(1, round(pil_image.width * scale)), max(1, round(pil_image.height * scale)))
    return pil_image.resize(new_size, PilImage.Resampling.BILINEAR)


def to_uint8_array(pil_image: PilImage.Image) -> np.ndarray:
    """Return a height-width-channel uint8 array."""
    return np.asarray(pil_image, dtype=np.uint8)

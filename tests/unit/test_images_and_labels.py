from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image as PilImage

from counter.adapters.detector.images import decode, resize_longest_side, to_uint8_array
from counter.adapters.detector.labels import class_name_for, load_labels
from counter.domain.errors import InvalidImageError, ModelLoadError
from counter.domain.models import Image

pytestmark = pytest.mark.unit


class TestDecode:
    def test_decodes_a_jpeg(self, image_bytes: bytes) -> None:
        decoded = decode(Image(content=image_bytes))

        assert decoded.mode == "RGB"
        assert to_uint8_array(decoded).shape[2] == 3

    def test_decodes_rgba_png(self, png_with_alpha: bytes) -> None:
        # The original adapter reshaped the raw buffer to (h, w, 3); a PNG with
        # an alpha channel has four bands and blew up with a ValueError.
        assert to_uint8_array(decode(Image(content=png_with_alpha))).shape == (24, 32, 3)

    def test_decodes_grayscale(self, grayscale_jpeg: bytes) -> None:
        assert to_uint8_array(decode(Image(content=grayscale_jpeg))).shape == (24, 32, 3)

    def test_rejects_an_empty_payload(self) -> None:
        with pytest.raises(InvalidImageError, match="empty"):
            decode(Image(content=b""))

    def test_rejects_a_non_image(self) -> None:
        with pytest.raises(InvalidImageError, match="not a decodable image"):
            decode(Image(content=b"this is a text file, not a jpeg"))

    def test_rejects_a_payload_over_the_limit(self, image_bytes: bytes) -> None:
        with pytest.raises(InvalidImageError, match="limit is"):
            decode(Image(content=image_bytes), max_bytes=10)

    def test_rejects_an_image_over_the_pixel_limit(self, image_bytes: bytes) -> None:
        with pytest.raises(InvalidImageError, match="pixel limit"):
            decode(Image(content=image_bytes), max_pixels=10)


class TestResize:
    def test_downscales_the_longest_side_and_keeps_the_aspect_ratio(self) -> None:
        resized = resize_longest_side(PilImage.new("RGB", (2000, 1000)), 500)

        assert resized.size == (500, 250)

    def test_leaves_small_images_alone(self) -> None:
        original = PilImage.new("RGB", (100, 80))

        assert resize_longest_side(original, 500) is original


class TestLabels:
    def test_loads_the_packaged_coco_map(self) -> None:
        labels = load_labels()

        assert labels[1] == "person"
        assert labels[17] == "cat"

    def test_loads_a_plain_text_map_by_path(self, tmp_path: Path) -> None:
        path = tmp_path / "classes.txt"
        path.write_text("person\nbicycle\ncar\n", encoding="utf-8")

        assert load_labels(str(path)) == {0: "person", 1: "bicycle", 2: "car"}

    def test_reports_a_missing_map_clearly(self) -> None:
        with pytest.raises(ModelLoadError, match="label map not found"):
            load_labels("no_such_labels.json")

    def test_reports_a_malformed_map_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text('[{"id": 1}]', encoding="utf-8")

        with pytest.raises(ModelLoadError, match="malformed"):
            load_labels(str(path))

    def test_resolves_float_class_ids(self) -> None:
        # TF Serving sends detection_classes as floats; the original code looked
        # them up in an int-keyed dict and raised KeyError on every detection.
        assert class_name_for({17: "cat"}, 17.0) == "cat"

    def test_unknown_ids_get_a_synthetic_name_instead_of_being_dropped(self) -> None:
        assert class_name_for({17: "cat"}, 999) == "class_999"

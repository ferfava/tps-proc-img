import cv2
import numpy as np
import pytest

from docscan import (
    ColorCorrectionFilter,
    DocumentDetector,
    DocumentNotFound,
    ScanOptions,
    Scanner,
    SoftenColorsFilter,
    ordenar_puntos,
)
from test_contract import document, no_document, photo


def test_order_corners_returns_clockwise_document_order():
    points = np.array([[20, 90], [100, 20], [20, 20], [100, 90]])

    ordered = ordenar_puntos(points)

    np.testing.assert_array_equal(
        ordered,
        [[20, 20], [100, 20], [100, 90], [20, 90]],
    )


@pytest.mark.parametrize(
    ("angle", "perspective"),
    [(0, 0.0), (12, 0.0), (-15, 0.0), (6, 0.12)],
)
def test_detector_finds_synthetic_document_corners(angle, perspective):
    rgb, expected = photo(document(), angle=angle, perspective=perspective)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    found = DocumentDetector().detect(image)

    errors = np.linalg.norm(found - expected, axis=1)
    assert np.max(errors) <= 0.02 * np.hypot(*image.shape[:2])


@pytest.mark.parametrize("kind", ["uniform", "circles", "tiny"])
def test_detector_raises_when_document_is_not_recognizable(kind):
    with pytest.raises(DocumentNotFound, match="cuatro bordes"):
        DocumentDetector().detect(no_document(kind))


def test_color_filters_preserve_neutral_pixels_and_lighten_red():
    image = np.array([[[0, 0, 0], [100, 100, 100], [255, 255, 255], [40, 30, 225]]], dtype=np.uint8)

    softened = SoftenColorsFilter(0.7).apply(image)

    np.testing.assert_array_equal(softened[0, :3], image[0, :3])
    assert softened[0, 3].mean() > image[0, 3].mean()


def test_color_correction_neutralizes_warm_paper():
    image = np.full((40, 40, 3), (160, 212, 235), dtype=np.uint8)
    image[:5, :5] = 0

    corrected = ColorCorrectionFilter().apply(image)

    np.testing.assert_array_equal(corrected[20, 20], [255, 255, 255])


@pytest.mark.parametrize("color_mode", ["color", "grayscale", "bw"])
def test_scanner_runs_pipeline_in_each_color_mode(color_mode):
    rgb, _ = photo(document(), angle=12)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = Scanner().scan(
        image,
        ScanOptions(color_mode=color_mode, color_correction=False),
    )

    assert result.corners.shape == (4, 2)
    assert result.image.ndim == (2 if color_mode != "color" else 3)
    if color_mode == "bw":
        assert set(np.unique(result.image)).issubset({0, 255})
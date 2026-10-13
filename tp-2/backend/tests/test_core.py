"""
Tests para verificar el funcionamiento del detector geométrico,
el corrector de perspectiva y el escáner principal.
"""

import numpy as np
import pytest

from docscan.detection import DocumentDetector, order_corners
from docscan.exceptions import DocumentNotFound
from docscan.perspective import PerspectiveCorrector
from docscan.scanner import Scanner


def test_order_corners_sorts_shuffled_points() -> None:
    """Prueba que order_corners ordene correctamente un conjunto de esquinas desordenadas."""
    # Esquinas de ejemplo mezcladas
    shuffled = np.array([
        [480, 700],
        [90, 680],
        [100, 100],
        [500, 120]
    ], dtype=np.float32)

    # Orden esperado: [arriba-izq, arriba-der, abajo-der, abajo-izq]
    expected = np.array([
        [100, 100],
        [500, 120],
        [480, 700],
        [90, 680]
    ], dtype=np.float32)

    result = order_corners(shuffled)
    assert np.allclose(result, expected, atol=2)


def test_detector_finds_white_sheet() -> None:
    """Prueba que DocumentDetector encuentre con precisión las esquinas de una hoja blanca sobre una mesa oscura."""
    # Mesa gris oscura de 400x400 con una hoja blanca en el centro
    image = np.full((400, 400, 3), 40, np.uint8)
    image[100:300, 80:320] = 255

    detector = DocumentDetector()
    result = detector.detect(image)

    expected = np.array([
        [80, 100],
        [320, 100],
        [320, 300],
        [80, 300]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=2)


def test_detector_raises_when_no_document() -> None:
    """Prueba que DocumentDetector lance DocumentNotFound si la foto es lisa y no hay hoja."""
    # Imagen totalmente lisa sin contraste ni contornos
    image = np.full((400, 400, 3), 128, np.uint8)
    detector = DocumentDetector()

    with pytest.raises(DocumentNotFound):
        detector.detect(image)


def test_perspective_keeps_rectangle_size() -> None:
    """Prueba que PerspectiveCorrector mantenga las dimensiones exactas al enderezar un rectángulo."""
    # Imagen sintética de 200x300
    image = np.full((200, 300, 3), 200, np.uint8)
    
    # Esquinas que delimitan un área de 100 de ancho (de 10 a 110) y 50 de alto (de 20 a 70)
    corners = np.array([
        [10, 20],
        [110, 20],
        [110, 70],
        [10, 70]
    ], dtype=np.float32)

    corrector = PerspectiveCorrector()
    result = corrector.warp(image, corners)

    # El shape en NumPy es (alto, ancho), por lo que esperamos (50, 100)
    assert result.shape[:2] == (50, 100)


def test_scanner_grayscale_returns_single_channel() -> None:
    """Prueba que el escáner completo devuelva una imagen de un solo canal al usar grayscale."""
    # Imagen de prueba con la hoja blanca
    image = np.full((400, 400, 3), 40, np.uint8)
    image[100:300, 80:320] = 255

    scanner = Scanner()
    scan_result = scanner.scan(image, color_mode="grayscale")

    # El resultado final procesado debe ser de un solo canal (ndim == 2)
    assert scan_result.image.ndim == 2


def test_scanner_rejects_unknown_color_mode() -> None:
    """Prueba que el escáner rechace un modo de color inexistente lanzando un ValueError."""
    image = np.full((400, 400, 3), 40, np.uint8)
    image[100:300, 80:320] = 255

    scanner = Scanner()
    with pytest.raises(ValueError):
        scanner.scan(image, color_mode="sepia")
        
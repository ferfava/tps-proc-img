"""
Tests para verificar el funcionamiento de los filtros de escaneo.
"""

import numpy as np
import pytest

from docscan.filters import BlackAndWhite, ColorCorrection, Grayscale, SoftenColors


def test_grayscale_returns_single_channel() -> None:
    """Prueba que el filtro Grayscale convierta una imagen a color en una de un solo canal."""
    image = np.full((10, 10, 3), (160, 212, 235), np.uint8)
    filter_obj = Grayscale()
    result = filter_obj.apply(image)
    assert result.ndim == 2


def test_grayscale_keeps_gray_image_as_copy() -> None:
    """Prueba que si la imagen ya es gris, Grayscale devuelva una copia y no el mismo objeto."""
    image = np.full((10, 10), 128, np.uint8)
    filter_obj = Grayscale()
    result = filter_obj.apply(image)
    assert result.ndim == 2
    assert result is not image


def test_black_and_white_has_only_black_and_white() -> None:
    """Prueba que BlackAndWhite binarice la imagen dejando únicamente valores 0 y 255."""
    image = np.zeros((10, 10, 3), np.uint8)
    image[:, 5:] = 200  # Pintamos de gris claro desde la columna 5 en adelante
    
    filter_obj = BlackAndWhite(block_size=3, c=5)
    result = filter_obj.apply(image)
    
    unique_values = set(np.unique(result).tolist())
    assert unique_values <= {0, 255}


def test_black_and_white_rejects_even_block_size() -> None:
    """Prueba que BlackAndWhite rechace un block_size par lanzando un ValueError."""
    with pytest.raises(ValueError):
        BlackAndWhite(block_size=30)


def test_color_correction_whitens_tinted_paper() -> None:
    """Prueba que ColorCorrection blanquee un papel amarillento haciendo que sus valores sean altos."""
    image = np.full((10, 10, 3), (160, 212, 235), np.uint8)
    filter_obj = ColorCorrection(percentile=95.0)
    result = filter_obj.apply(image)
    assert result.min() >= 250


def test_color_correction_keeps_ink_dark() -> None:
    """Prueba que ColorCorrection mantenga la tinta oscura intacta en el papel corregido."""
    image = np.full((10, 10, 3), (160, 212, 235), np.uint8)
    image[:3, :3] = 25  # Cuadradito de tinta oscura
    filter_obj = ColorCorrection(percentile=95.0)
    result = filter_obj.apply(image)
    assert result[0, 0].max() < 60


def test_soften_colors_zero_changes_nothing() -> None:
    """Prueba que SoftenColors con fuerza 0 no altere la imagen original."""
    image = np.full((10, 10, 3), (100, 150, 200), np.uint8)
    filter_obj = SoftenColors(strength=0.0)
    result = filter_obj.apply(image)
    assert np.array_equal(result, image)


def test_soften_colors_does_not_change_black() -> None:
    """Prueba que SoftenColors con fuerza 1 no altere una imagen totalmente negra."""
    image = np.zeros((10, 10, 3), np.uint8)
    filter_obj = SoftenColors(strength=1.0)
    result = filter_obj.apply(image)
    assert np.array_equal(result, image)


def test_soften_colors_lightens_strong_red() -> None:
    """Prueba que SoftenColors aclare un rojo fuerte manteniendo su predominio cromático."""
    # En BGR el rojo es (azul=40, verde=30, rojo=225)
    image = np.full((10, 10, 3), (40, 30, 225), np.uint8)
    filter_obj = SoftenColors(strength=1.0)
    result = filter_obj.apply(image)
    
    # (a) El verde (canal 1) tiene que subir (más que 30)
    assert result[0, 0, 1] > 30
    # (b) El rojo (canal 2) tiene que seguir siendo el mayor de los tres canales
    assert result[0, 0, 2] > result[0, 0, 0] and result[0, 0, 2] > result[0, 0, 1]


def test_soften_colors_rejects_strength_out_of_range() -> None:
    """Prueba que SoftenColors rechace un valor de strength fuera de rango (ej. 1.5)."""
    with pytest.raises(ValueError):
        SoftenColors(strength=1.5)
        
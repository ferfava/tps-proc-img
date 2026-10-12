"""
Acá están los filtros que mejoran el escaneo.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Filter(ABC):
    """
    Esta es una clase molde (abstracta) para todos los filtros de imágenes.
    No se puede usar directamente, obliga a las otras clases a tener un método 'apply'.
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Devuelve una imagen nueva con el filtro aplicado."""
        pass


class Grayscale(Filter):
    """
    Filtro que convierte una imagen a escala de grises.
    Si ya está en grises, la devuelve tal cual para evitar errores.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica la conversión de color BGR a Gris usando OpenCV."""
        # Si la imagen ya tiene 2 dimensiones, significa que ya es gris. La devuelvo copiada.
        if image.ndim == 2:
            return image.copy()

        # Si no, la paso de color (BGR) a grises usando OpenCV
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class BlackAndWhite(Filter):
    """
    Filtro que convierte la imagen a blanco y negro puro (binaria)
    usando umbral adaptativo para lidiar con las sombras.
    """

    def __init__(self, block_size: int = 31, c: int = 15) -> None:
        """Guarda y valida los parámetros del umbral adaptativo."""
        # El tamaño del bloque debe ser un número impar mayor o igual a 3 para OpenCV
        if block_size % 2 == 0 or block_size < 3:
            raise ValueError("El block_size debe ser un número impar mayor o igual a 3.")

        self.block_size = block_size
        self.c = c

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Pasa la imagen a grises y aplica umbral adaptativo."""
        # 1. Primero la pasamos a grises usando el filtro que ya armamos (¡sin repetir código!)
        gray_image = Grayscale().apply(image)

        # 2. Después aplicamos el umbral adaptativo de OpenCV para lidiar con las sombras
        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )
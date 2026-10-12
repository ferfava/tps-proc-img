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
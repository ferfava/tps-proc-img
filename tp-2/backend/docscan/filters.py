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
        # Si la imagen ya tiene 2 dimensiones, significa que ya es gris.
        # La devuelvo copiada.
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
        # 1. Primero la pasamos a grises usando el filtro
        # que ya armamos (¡sin repetir código!)
        gray_image = Grayscale().apply(image)

        # 2. Después aplicamos el umbral adaptativo de OpenCV
        # para lidiar con las sombras
        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )


class ColorCorrection(Filter):
    """
    Filtro que corrige el balance de blancos de la imagen para que el papel
    amarillento por la luz cálida vuelva a verse blanco.
    """

    def __init__(self, percentile: float = 95.0) -> None:
        """Guarda el percentil que se usará para detectar el color del papel."""
        self.percentile = percentile

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica la corrección de color estirando los canales según el percentil."""
        # Si la imagen es en grises (2 dimensiones), no tiene canales de
        # color para corregir, la devolvemos igual
        if image.ndim == 2:
            return image.copy()

        # a. Calculamos el color del papel usando el percentil para cada canal (B, G, R)
        paper_color = np.percentile(image, self.percentile, axis=(0, 1))

        # b. Calculamos cuánto estirar cada color (255 / color del papel). 
        # Usamos np.maximum para evitar dividir por 0 en caso de
        # que alguna parte sea negra pura.
        scaling_factors = 255.0 / np.maximum(paper_color, 1)

        # c. Pasamos la imagen a decimales (float32) y la multiplicamos 
        # por los factores (broadcasting)
        corrected_image = image.astype(np.float32) * scaling_factors

        # d. Recortamos con np.clip para que ningún valor pase de 255
        # ni baje de 0, y volvemos a uint8
        corrected_image = np.clip(corrected_image, 0, 255)
        
        return corrected_image.astype(np.uint8)


class SoftenColors(Filter):
    """
    Filtro que aclara los colores fuertes (como resaltadores o sellos)
    mezclándolos sutilmente con el blanco usando el espacio HSV.
    """

    def __init__(self, strength: float = 0.0) -> None:
        """Guarda y valida la fuerza del suavizado de colores."""
        if not (0.0 <= strength <= 1.0):
            raise ValueError("El strength debe estar entre 0.0 y 1.0.")
        self.strength = strength

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aclara colores intensos evaluando su saturación y brillo en HSV."""
        # Si la imagen es en grises (2 dimensiones), no tiene colores, la devolvemos igual
        if image.ndim == 2:
            return image.copy()

        # 1. Convertimos la imagen de BGR a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Extraemos las capas de Saturación (S) y Brillo (V) y las pasamos a float32
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        # 3. Calculamos el peso de aclarado para cada píxel
        weight = self.strength * (s / 255.0) * (v / 255.0)

        # 4. Pasamos la imagen a decimales para operar sin perder precisión
        image_float = image.astype(np.float32)

        # 5. Aplicamos la mezcla con el blanco usando np.newaxis para el broadcasting
        weight_expanded = weight[:, :, np.newaxis]
        softened_image = image_float + weight_expanded * (255.0 - image_float)

        # 6. Recortamos los valores entre 0 y 255, y volvemos a formato uint8
        softened_image = np.clip(softened_image, 0, 255)
        return softened_image.astype(np.uint8)
"""
Módulo para la detección y el procesamiento geométrico de las esquinas del documento.
"""

import cv2
import numpy as np

from docscan.exceptions import DocumentNotFound
from docscan.filters import Grayscale


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Recibe un array con 4 esquinas desordenadas (x, y) y las devuelve 
    ordenadas estrictamente en el siguiente orden:
    [arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda].
    """
    # 1. Calculamos la suma de (x + y) para cada fila
    sums = points.sum(axis=1)
    
    # 2. Calculamos la diferencia (y - x) para cada fila y aplanamos con .ravel()
    diffs = np.diff(points, axis=1).ravel()

    # 3. Identificamos cada esquina según sus valores extremos
    top_left = points[np.argmin(sums)]      # La suma más chica está cerca del origen (0,0)
    bottom_right = points[np.argmax(sums)]  # La suma más grande está más alejada
    top_right = points[np.argmin(diffs)]    # La diferencia y - x más chica (x grande, y chico)
    bottom_left = points[np.argmax(diffs)]  # La diferencia y - x más grande (x chico, y grande)

    # 4. Devolvemos el array ordenado como pide el estándar
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


class DocumentDetector:
    """
    Clase encargada de procesar una imagen para encontrar las 4 esquinas 
    del documento principal utilizando técnicas de visión por computadora.
    """

    def __init__(
        self,
        blur_ksize: int = 5,
        canny_threshold1: int = 75,
        canny_threshold2: int = 200,
        min_area_ratio: float = 0.15,
        approx_epsilon: float = 0.02,
        close_kernel_size: int = 5,
        work_size: int = 600
    ) -> None:
        """Guarda los parámetros de configuración para la detección de bordes."""
        self.blur_ksize = blur_ksize
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon = approx_epsilon
        self.close_kernel_size = close_kernel_size
        self.work_size = work_size

    def _find_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica una serie de preprocesamientos para encontrar los bordes de la imagen:
        1. Convierte a grises reutilizando el filtro Grayscale.
        2. Aplica desenfoque gaussiano para reducir el ruido.
        3. Detecta bordes utilizando el algoritmo de Canny.
        4. Cierra los huecos en los bordes mediante operaciones morfológicas (Closing).
        """
        # 1. Pasar a grises usando el filtro que ya armamos en el módulo filters
        gray = Grayscale().apply(image)

        # 2. Desenfoque gaussiano para suavizar la imagen (el ksize debe ser impar, ej. 5x5)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # 3. Detección de bordes con Canny usando los umbrales configurados
        edged = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        # 4. Cerrar huecos en los bordes usando morfología matemática (Dilation seguida de Erosion)
        kernel = np.ones((self.close_kernel_size, self.close_kernel_size), np.uint8)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        return closed

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta el documento principal en la imagen, encontrando y ordenando 
        sus 4 esquinas para su posterior corrección geométrica.
        """
        # Parte A: Achicar la foto para agilizar el procesamiento
        height, width = image.shape[:2]
        scale = self.work_size / max(height, width)
        new_width = round(width * scale)
        new_height = round(height * scale)
        small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Parte B: Obtener bordes y contornos
        edges = self._find_edges(small)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Área mínima requerida (proporción respecto al tamaño de la imagen de trabajo)
        min_area = self.min_area_ratio * small.shape[0] * small.shape[1]

        # Parte C: Buscar la hoja entre los contornos (ordenados de mayor a menor área)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * perimeter, True)

            # Validamos que tenga 4 lados, sea convexo y su área sea suficiente
            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and cv2.contourArea(approx) >= min_area
            ):
                # Pasamos a float32, reajustamos a la forma (4, 2) y reescalamos a la resolución original
                points = approx.reshape(4, 2).astype(np.float32)
                points = points / scale
                return order_corners(points)

        # Parte D: Si no se encuentra un contorno válido, lanzamos la excepción personalizada
        raise DocumentNotFound(
            "No se pudo detectar el documento. Asegurate de que "
            "las 4 esquinas sean visibles, el fondo contraste con la hoja y haya buena iluminación."
        )
        
"""
Módulo para la detección y el procesamiento geométrico de las esquinas del documento.
"""

import numpy as np


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
        
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
    
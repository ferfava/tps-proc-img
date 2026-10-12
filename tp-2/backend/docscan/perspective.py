"""
Módulo para la corrección geométrica de perspectiva y recorte de documentos.
"""

import cv2
import numpy as np


class PerspectiveCorrector:
    """
    Clase encargada de aplicar una transformación de perspectiva a la imagen
    para enderezar y recortar el documento basándose en sus 4 esquinas detectadas.
    """

    def warp(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Recibe la imagen original y las 4 esquinas ordenadas del documento,
        calcula las dimensiones reales conservando proporciones, y devuelve
        la hoja enderezada de frente mediante un 'warp perspective'.
        """
        # a. Desempaquetamos las 4 esquinas ordenadas en variables individuales
        top_left, top_right, bottom_right, bottom_left = corners

        # b. Calculamos el ancho de la hoja como el mayor entre el borde superior e inferior
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        width = int(round(max(width_top, width_bottom)))

        # c. Calculamos el alto de la hoja como el mayor entre el borde izquierdo y derecho
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        height = int(round(max(height_left, height_right)))

        # d. Armamos el array con las coordenadas destino en el mismo orden (arriba-izq, arriba-der, abajo-der, abajo-izq)
        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ],
            dtype=np.float32
        )

        # e. Calculamos la matriz de transformación de perspectiva (la "receta")
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)

        # f. Aplicamos la transformación a la imagen original y la devolvemos (ancho, alto)
        warped = cv2.warpPerspective(image, matrix, (width, height))

        return warped
        
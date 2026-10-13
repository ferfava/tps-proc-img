import cv2
import numpy as np


class PerspectiveCorrector:
    """Corrige la perspectiva de un documento detectado."""

    def correct(
        self,
        image: np.ndarray,
        corners: np.ndarray
    ) -> np.ndarray:
        """Devuelve el documento enderezado y recortado."""
        
        points = np.asarray(corners, dtype=np.float32)

        top_left, top_right, bottom_right, bottom_left = points

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)

        width = int(round(max(width_top, width_bottom)))

        
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        height = int(round(max(height_left, height_right)))

        
        destination = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        
        if width < 2 or height < 2:
            raise ValueError("Las dimensiones del documento no son válidas.")

        matrix = cv2.getPerspectiveTransform(points, destination)

        corrected_image = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR
        )

        return corrected_image

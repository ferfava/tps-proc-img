
import numpy as np

from docscan.detect import detect_document
from docscan.perspective import PerspectiveCorrector


class DocumentScanner:
    """Coordina la detección y la corrección de perspectiva."""

    def __init__(self) -> None:
        self.corrector = PerspectiveCorrector()

    def scan(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve la imagen corregida y las esquinas detectadas."""
        corners = np.asarray(
            detect_document(image),
            dtype=np.float32
        )

        corrected_image = self.corrector.correct(image, corners)

        return corrected_image, corners

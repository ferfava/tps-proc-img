"""
Acá están los filtros que mejoran el escaneo.
"""

# Traigo de la librería de Python lo necesario para hacer clases abstractas (moldes)
from abc import ABC, abstractmethod

# Traigo numpy para manejar la imagen como si fuera una matriz de números
import numpy as np


# Creo la clase Filter que usa ABC para que sea un molde y no se use directamente
class Filter(ABC):
    """
    Esta es una clase molde (abstracta) para todos los filtros de imágenes.
    No se puede usar directamente, obliga a las otras clases a tener un método 'apply'.
    """

    # Obligo a las clases hijas a definir este método
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Devuelve una imagen nueva con el filtro aplicado."""
        # No pongo código acá porque es solo la estructura del molde
        pass
"""
Módulo principal del escáner, encargado de coordinar la detección geométrica,
la corrección de perspectiva y la aplicación secuencial de filtros.
"""

from dataclasses import dataclass

import numpy as np

from docscan.detection import DocumentDetector
from docscan.filters import (
    BlackAndWhite,
    ColorCorrection,
    Filter,
    Grayscale,
    SoftenColors,
)
from docscan.perspective import PerspectiveCorrector

# Diccionario para asociar cada modo de color con su respectiva clase de filtro (sin if/elif)
COLOR_MODE_FILTERS = {
    "color": None,
    "grayscale": Grayscale,
    "bw": BlackAndWhite,
}


@dataclass(frozen=True)
class ScanResult:
    """
    Cajita inmutable de datos que almacena la imagen final escaneada
    y las esquinas detectadas del documento.
    """

    image: np.ndarray
    corners: np.ndarray


class Scanner:
    """
    Director de orquesta del módulo docscan. Coordina la detección de esquinas,
    la perspectiva y la cadena de filtros configurada.
    """

    def __init__(
        self,
        detector: DocumentDetector | None = None,
        corrector: PerspectiveCorrector | None = None
    ) -> None:
        """Inicializa el escáner con el detector y el corrector provistos o por defecto."""
        self.detector = detector or DocumentDetector()
        self.corrector = corrector or PerspectiveCorrector()

    def _build_filters(
        self,
        color_mode: str,
        color_correction: bool,
        soften_colors: float
    ) -> list[Filter]:
        """Construye y devuelve la lista ordenada de filtros a aplicar según la configuración."""
        filters: list[Filter] = []

        # 1. Si está activo, agregamos primero la corrección de balance de blancos
        if color_correction:
            filters.append(ColorCorrection())

        # 2. Si la fuerza de suavizado es mayor a 0, agregamos el filtro de resaltadores/sellos
        if soften_colors > 0.0:
            filters.append(SoftenColors(strength=soften_colors))

        # 3. Validamos que el modo de color exista en el diccionario para evitar errores silenciosos
        if color_mode not in COLOR_MODE_FILTERS:
            raise ValueError(
                f"Modo de color inválido: '{color_mode}'. "
                f"Opciones válidas: {list(COLOR_MODE_FILTERS.keys())}"
            )

        # 4. Agregamos el filtro correspondiente al modo de color elegido
        filter_class = COLOR_MODE_FILTERS[color_mode]
        if filter_class is not None:
            filters.append(filter_class())

        return filters

    def scan(
        self,
        image: np.ndarray,
        color_mode: str = "color",
        color_correction: bool = True,
        soften_colors: float = 0.0
    ) -> ScanResult:
        """
        Ejecuta el pipeline completo de escaneo:
        1. Detecta las 4 esquinas del documento.
        2. Endereza y recorta la hoja mediante transformación de perspectiva.
        3. Aplica de forma polimórfica la cadena de filtros seleccionada.
        """
        # 1. Obtenemos las esquinas del documento mediante el detector
        corners = self.detector.detect(image)

        # 2. Enderezamos y recortamos la hoja con el corrector de perspectiva
        warped_image = self.corrector.warp(image, corners)

        # 3. Construimos la lista de filtros y los aplicamos secuencialmente (polimorfismo)
        filters = self._build_filters(color_mode, color_correction, soften_colors)
        
        processed_image = warped_image
        for filter_obj in filters:
            processed_image = filter_obj.apply(processed_image)

        # 4. Devolvemos el resultado empaquetado en la dataclass inmutable
        return ScanResult(image=processed_image, corners=corners)
        
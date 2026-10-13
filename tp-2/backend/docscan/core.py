"""Reusable document detection and scanning pipeline built on OpenCV."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from docscan.exceptions import DocumentNotFound, InvalidImage


def _validate_image(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise InvalidImage("La imagen debe ser un array BGR válido de 8 bits.")


def ordenar_puntos(points: np.ndarray) -> np.ndarray:
    """Devuelve cuatro puntos como sup. izq., sup. der., inf. der., inf. izq."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2):
        raise ValueError("Se necesitan exactamente cuatro puntos.")

    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


class DocumentDetector:
    """Encuentra el cuadrilátero convexo que representa al documento."""

    def __init__(
        self,
        min_area_ratio: float = 0.15,
        max_working_side: int = 1000,
        canny_thresholds: tuple[int, int] = (50, 150),
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_working_side = max_working_side
        self.canny_thresholds = canny_thresholds

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Retorna las esquinas en píxeles de la imagen original."""
        _validate_image(image)
        height, width = image.shape[:2]
        scale = min(1.0, self.max_working_side / max(height, width))
        if scale < 1.0:
            working = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, *self.canny_thresholds)
        kernel = np.ones((5, 5), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        working_area = working.shape[0] * working.shape[1]
        minimum_area = working_area * self.min_area_ratio
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                break
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue

            corners = ordenar_puntos(approximation)
            if self._covers_image_border(corners, working.shape[1], working.shape[0]):
                continue
            if scale < 1.0:
                corners /= scale
            return corners.astype(np.float32)

        raise DocumentNotFound(
            "No se encontró un documento claro en la foto. Asegurate de que se "
            "vean sus cuatro bordes, ocupe una parte importante de la imagen y "
            "contraste con el fondo."
        )

    @staticmethod
    def _covers_image_border(corners: np.ndarray, width: int, height: int) -> bool:
        min_x, min_y = corners.min(axis=0)
        max_x, max_y = corners.max(axis=0)
        return (
            min_x <= 1
            and min_y <= 1
            and max_x >= width - 2
            and max_y >= height - 2
        )


class PerspectiveCorrector:
    """Recorta el documento y lo rectifica sin fijar una proporción."""

    def correct(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        _validate_image(image)
        source = ordenar_puntos(corners)
        top_left, top_right, bottom_right, bottom_left = source

        output_width = max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
        output_height = max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
        width = max(1, round(float(output_width)))
        height = max(1, round(float(output_height)))
        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(image, transform, (width, height))


class Filter(ABC):
    """Interfaz común para los filtros del escáner."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica el filtro y devuelve una imagen nueva."""


class ColorCorrectionFilter(Filter):
    """Neutraliza la dominante de color usando percentiles altos por canal."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        corrected = image.astype(np.float32)
        white_points = np.percentile(corrected, 95, axis=(0, 1))
        white_points = np.maximum(white_points, 1.0)
        return np.clip(corrected * (255.0 / white_points), 0, 255).astype(np.uint8)


class SoftenColorsFilter(Filter):
    """Aclara colores saturados sin modificar píxeles grises ni oscuros."""

    def __init__(self, amount: float) -> None:
        if not 0.0 <= amount <= 1.0:
            raise ValueError("soften_colors debe estar entre 0 y 1.")
        self.amount = amount

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.amount == 0.0:
            return image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        brightness = np.clip((hsv[:, :, 2] - 80.0) / 175.0, 0.0, 1.0)
        strength = self.amount * 0.65 * saturation * brightness
        result = image.astype(np.float32)
        result = result * (1.0 - strength[:, :, None]) + 255.0 * strength[:, :, None]
        return np.clip(result, 0, 255).astype(np.uint8)


class GrayscaleFilter(Filter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class BlackAndWhiteFilter(Filter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smallest_side = min(gray.shape)
        if smallest_side < 3:
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return result
        block_size = min(31, smallest_side if smallest_side % 2 else smallest_side - 1)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            10,
        )


@dataclass(frozen=True)
class ScanOptions:
    color_mode: str = "color"
    color_correction: bool = True
    soften_colors: float = 0.0

    def __post_init__(self) -> None:
        if self.color_mode not in {"color", "grayscale", "bw"}:
            raise ValueError("color_mode debe ser color, grayscale o bw.")
        if not 0.0 <= self.soften_colors <= 1.0:
            raise ValueError("soften_colors debe estar entre 0 y 1.")


@dataclass(frozen=True)
class ScanResult:
    image: np.ndarray
    corners: np.ndarray


class Scanner:
    """Coordina detección, corrección de perspectiva y filtros."""

    def __init__(
        self,
        detector: DocumentDetector | None = None,
        corrector: PerspectiveCorrector | None = None,
    ) -> None:
        self.detector = detector or DocumentDetector()
        self.corrector = corrector or PerspectiveCorrector()

    def scan(
        self, image: np.ndarray, options: ScanOptions | None = None
    ) -> ScanResult:
        options = options or ScanOptions()
        corners = self.detector.detect(image)
        result = self.corrector.correct(image, corners)
        filters = self._build_filters(options)
        for image_filter in filters:
            result = image_filter.apply(result)
        return ScanResult(image=result, corners=corners)

    @staticmethod
    def _build_filters(options: ScanOptions) -> list[Filter]:
        filters: list[Filter] = []
        if options.color_correction:
            filters.append(ColorCorrectionFilter())
        if options.soften_colors > 0.0:
            filters.append(SoftenColorsFilter(options.soften_colors))
        if options.color_mode == "grayscale":
            filters.append(GrayscaleFilter())
        elif options.color_mode == "bw":
            filters.append(BlackAndWhiteFilter())
        return filters


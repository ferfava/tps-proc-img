"""
docscan: document scanning core.

Turns a photo of a document into a clean, straight scan. It must NOT import
FastAPI nor anything from `app`, so it can be used from the API, from the
command line (`python -m docscan`) or from any other Python program.

Given: image input/output (imageio.py) and the domain exceptions (exceptions.py).
Detection, perspective correction, filters and the scanner live in core.py.
TODO (team): the command line (__main__.py).
"""

from docscan.core import (
    BlackAndWhiteFilter,
    ColorCorrectionFilter,
    DocumentDetector,
    Filter,
    GrayscaleFilter,
    PerspectiveCorrector,
    ScanOptions,
    ScanResult,
    Scanner,
    SoftenColorsFilter,
    ordenar_puntos,
)
from docscan.exceptions import DocumentNotFound, InvalidImage, ScanError
from docscan.imageio import Photo, encode_png, from_pil, read_photo, to_pil

__all__ = [
    "BlackAndWhiteFilter",
    "ColorCorrectionFilter",
    "DocumentDetector",
    "DocumentNotFound",
    "Filter",
    "GrayscaleFilter",
    "InvalidImage",
    "PerspectiveCorrector",
    "Photo",
    "ScanOptions",
    "ScanResult",
    "ScanError",
    "Scanner",
    "SoftenColorsFilter",
    "encode_png",
    "from_pil",
    "ordenar_puntos",
    "read_photo",
    "to_pil",
]

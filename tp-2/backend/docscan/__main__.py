
"""Interfaz de línea de comandos del escáner de documentos."""

import argparse
import sys
from pathlib import Path

from docscan.exceptions import ScanError
from docscan.filters import (
    BWFilter,
    Color,
    ColorCorrectionFilter,
    GrayscaleFilter,
    SoftenColorsFilter,
)
from docscan.imageio import encode_png, read_photo
from docscan.scanner import DocumentScanner


def main() -> int:
    """Ejecuta el escáner desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Escáner de documentos con OpenCV"
    )

    parser.add_argument("entrada", type=Path)
    parser.add_argument("salida", type=Path)

    parser.add_argument(
        "--color-mode",
        choices=["color", "grayscale", "bw"],
        default="color",
    )

    parser.add_argument(
        "--no-color-correction",
        action="store_true",
    )

    parser.add_argument(
        "--soften-colors",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    if not 0.0 <= args.soften_colors <= 1.0:
        parser.error("--soften-colors debe estar entre 0 y 1")

    try:
        photo = read_photo(args.entrada.read_bytes())

        scanner = DocumentScanner()
        image, _ = scanner.scan(photo.image)

        if not args.no_color_correction:
            image = ColorCorrectionFilter().apply(image)

        if args.soften_colors > 0:
            image = SoftenColorsFilter(
                args.soften_colors
            ).apply(image)

        filters = {
            "color": Color(),
            "grayscale": GrayscaleFilter(),
            "bw": BWFilter(),
        }

        image = filters[args.color_mode].apply(image)

        png_content = encode_png(image)

        args.salida.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.salida.write_bytes(png_content)

        print(f"Escaneo guardado en: {args.salida}")
        return 0

    except (ScanError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

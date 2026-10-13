"""Command-line interface for the document scanner."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from docscan.core import ScanOptions, Scanner
from docscan.exceptions import ScanError
from docscan.imageio import encode_png, read_photo


def _soften_colors_value(value: str) -> float:
    try:
        amount = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("debe ser un número entre 0 y 1") from exc
    if not 0.0 <= amount <= 1.0:
        raise argparse.ArgumentTypeError("debe ser un número entre 0 y 1")
    return amount


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m docscan",
        description="Detecta y procesa un documento fotografiado.",
    )
    parser.add_argument("input", type=Path, help="foto de entrada")
    parser.add_argument("output", type=Path, help="archivo PNG de salida")
    parser.add_argument(
        "--color-mode",
        choices=("color", "grayscale", "bw"),
        default="color",
        help="modo de color del escaneo (predeterminado: color)",
    )
    parser.add_argument(
        "--no-color-correction",
        action="store_false",
        dest="color_correction",
        help="desactiva la corrección de color",
    )
    parser.add_argument(
        "--soften-colors",
        type=_soften_colors_value,
        default=0.0,
        metavar="0..1",
        help="intensidad para aclarar colores fuertes (predeterminado: 0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    options = ScanOptions(
        color_mode=args.color_mode,
        color_correction=args.color_correction,
        soften_colors=args.soften_colors,
    )

    try:
        photo = read_photo(args.input.read_bytes())
        result = Scanner().scan(photo.image, options)
        png = encode_png(result.image)
    except (OSError, ScanError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        args.output.write_bytes(png)
    except OSError as exc:
        print(f"Error al escribir {args.output}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
Módulo de la Interfaz de Línea de Comandos (CLI) para ejecutar el escáner
directamente desde la terminal utilizando argparse.
"""

import argparse
import sys
from pathlib import Path

from docscan.exceptions import ScanError
from docscan.imageio import encode_png, read_photo
from docscan.scanner import COLOR_MODE_FILTERS, Scanner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Configura y parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Escáner automático de documentos por visión por computadora."
    )
    
    # Argumentos posicionales obligatorios
    parser.add_argument(
        "input",
        type=Path,
        help="Ruta de la imagen de entrada a escanear (ej. foto.jpg)."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Ruta donde se guardará la imagen resultante escaneada (ej. escaneo.png)."
    )

    # Opciones de configuración
    parser.add_argument(
        "--color-mode",
        choices=list(COLOR_MODE_FILTERS.keys()),
        default="color",
        help="Modo de color del escaneo ('color', 'grayscale' o 'bw')."
    )
    parser.add_argument(
        "--no-color-correction",
        action="store_true",
        help="Desactiva la corrección automática del balance de blancos del papel."
    )
    parser.add_argument(
        "--soften-colors",
        type=float,
        default=0.0,
        help="Fuerza del suavizado para aclarar colores fuertes (entre 0.0 y 1.0)."
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Función principal de ejecución de la CLI.
    Maneja el flujo de lectura, escaneo, escritura y control robusto de errores.
    """
    args = parse_args(argv)

    try:
        # 1. Leemos los bytes del archivo de entrada y cargamos la foto
        image_bytes = args.input.read_bytes()
        photo = read_photo(image_bytes)

        # 2. Instanciamos el escáner y procesamos la imagen
        scanner = Scanner()
        
        # Ojo: la opción de la CLI es --no-color-correction, por lo que invertimos el booleano
        color_correction = not args.no_color_correction

        result = scanner.scan(
            image=photo.image,
            color_mode=args.color_mode,
            color_correction=color_correction,
            soften_colors=args.soften_colors
        )

        # 3. Codificamos el resultado a PNG y lo guardamos en la ruta de salida
        output_bytes = encode_png(result.image)
        args.output.write_bytes(output_bytes)

        return 0

    except (ScanError, ValueError, OSError) as error:
        # Si ocurre cualquier error esperado (detección fallida, modo inválido, archivo no encontrado),
        # lo imprimimos por stderr y terminamos con código de error (1) sin crear salida.
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
"""
Tests de integración para verificar el comportamiento del escáner
utilizando fotografías reales de documentos y casos de borde.
"""

from pathlib import Path

import pytest

from docscan.exceptions import DocumentNotFound
from docscan.imageio import read_photo
from docscan.scanner import Scanner

# Ruta a la carpeta de fotos de prueba ubicada junto a los tests
PHOTOS_DIR = Path(__file__).parent / "photos"


@pytest.mark.parametrize("name", ["hoja.jpg", "hoja_resaltador.jpg"])
def test_scans_real_documents(name: str) -> None:
    """Prueba que el escáner procese exitosamente fotos reales de documentos devolviendo una imagen legible."""
    photo_path = PHOTOS_DIR / name
    photo = read_photo(photo_path.read_bytes())

    # Ejecutamos el pipeline completo de escaneo
    result = Scanner().scan(photo.image)

    # Verificamos que la imagen resultante mantenga un tamaño legible (alto y ancho mayores a 500 píxeles)
    height, width = result.image.shape[:2]
    assert height > 500
    assert width > 500


@pytest.mark.parametrize("name", ["vaso.jpg", "hoja_cortada.jpg"])
def test_rejects_photos_without_document(name: str) -> None:
    """Prueba que el escáner lance DocumentNotFound cuando la imagen no contiene un documento válido."""
    photo_path = PHOTOS_DIR / name
    photo = read_photo(photo_path.read_bytes())

    # hoja_cortada.jpg no se reconoce porque no se logran ver las 4 esquinas completas del documento,
    # y el enunciado establece que en ese caso el sistema debe rechazarla explícitamente.
    with pytest.raises(DocumentNotFound):
        Scanner().scan(photo.image)
        
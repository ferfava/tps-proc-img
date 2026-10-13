
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.config import STORAGE_DIR
from app.schemas import ColorMode, ErrorOut, ScanOptions, ScanOut
from app.uploads import read_upload

from docscan.filters import (
    GrayscaleFilter,
    BWFilter,
    Color,
    ColorCorrectionFilter,
    SoftenColorsFilter,
)
from docscan.imageio import encode_png
from docscan.scanner import DocumentScanner


router = APIRouter(prefix="/api/scans", tags=["scans"])


def empty_scan(
    scan_id: str = "",
    original_name: str = "",
    options: ScanOptions | None = None,
) -> ScanOut:
    """Respuesta temporal para endpoints aún no implementados."""
    return ScanOut(
        id=scan_id,
        original_name=original_name,
        url="",
        original_url="",
        width=0,
        height=0,
        size_bytes=0,
        corners=[],
        options=options or ScanOptions(),
        created_at=datetime.now(),
    )


@router.post(
    "",
    response_model=ScanOut,
    status_code=status.HTTP_201_CREATED,
    summary="Scan a photo and save the result",
    responses={
        400: {"model": ErrorOut},
        413: {"model": ErrorOut},
        422: {"model": ErrorOut},
    },
)
async def create_scan(
    file: Annotated[
        UploadFile,
        File(description="Photo of a document")
    ],
    color_mode: Annotated[ColorMode, Form()] = "color",
    color_correction: Annotated[bool, Form()] = True,
    soften_colors: Annotated[float, Form(ge=0, le=1)] = 0.0,
):
    """Detecta, endereza, mejora y guarda un documento."""

    options = ScanOptions(
        color_mode=color_mode,
        color_correction=color_correction,
        soften_colors=soften_colors,
    )

    # 1. Leer y validar la imagen recibida.
    upload = await read_upload(file)

    # 2. Detectar las esquinas y corregir la perspectiva.
    # Si no hay documento, se lanza DocumentNotFound.
    scanner = DocumentScanner()

    new_image, corners = scanner.scan(upload.photo.image)

    # 3. Aplicar las mejoras de imagen.
    if color_correction:
        new_image = ColorCorrectionFilter().apply(new_image)

    if soften_colors > 0:
        new_image = SoftenColorsFilter(
            soften_colors
        ).apply(new_image)

    color_filters = {
        "color": Color(),
        "grayscale": GrayscaleFilter(),
        "bw": BWFilter(),
    }

    selected_filter = color_filters[color_mode]
    new_image = selected_filter.apply(new_image)

    # 4. Codificar el resultado como PNG.
    image_png = encode_png(new_image)

    # 5. Generar identificador único.
    scan_id = uuid.uuid4().hex

    # 6. Guardar la imagen corregida.
    (STORAGE_DIR / f"{scan_id}.png").write_bytes(image_png)

    # 7. Conservar la fotografía original sin alteraciones.
    extension = upload.photo.format.lower()

    original_filename = f"{scan_id}-original.{extension}"

    (STORAGE_DIR / original_filename).write_bytes(
        upload.content
    )

    # 8. Crear los metadatos del escaneo.
    scan = ScanOut(
        id=scan_id,
        original_name=upload.name,
        url=f"/media/{scan_id}.png",
        original_url=f"/media/{original_filename}",
        width=new_image.shape[1],
        height=new_image.shape[0],
        size_bytes=len(image_png),
        corners=corners.tolist(),
        options=options,
        created_at=datetime.now(),
    )

    # 9. Guardar metadatos JSON.
    (STORAGE_DIR / f"{scan_id}.json").write_text(
        scan.model_dump_json(),
        encoding="utf-8",
    )

    return scan


@router.get(
    "",
    response_model=list[ScanOut],
    summary="List the scans, newest first",
)
def list_scans():
    # Pendiente: implementar repositorio de almacenamiento.
    return []


@router.get(
    "/{scan_id}",
    response_model=ScanOut,
    summary="Get a scan",
    responses={404: {"model": ErrorOut}},
)
def get_scan(scan_id: str):
    # Pendiente: recuperar un escaneo por su identificador.
    return empty_scan(scan_id)


@router.delete(
    "/{scan_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a scan and its files",
    responses={404: {"model": ErrorOut}},
)
def delete_scan(scan_id: str):
    # Pendiente: eliminar imágenes y metadatos.
    return None

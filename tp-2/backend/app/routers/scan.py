
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.repository import ScanRepository
from app.schemas import ColorMode, ErrorOut, ScanOptions, ScanOut
from app.uploads import read_upload

from docscan.filters import (
    BWFilter,
    Color,
    ColorCorrectionFilter,
    GrayscaleFilter,
    SoftenColorsFilter,
)
from docscan.imageio import encode_png
from docscan.scanner import DocumentScanner


router = APIRouter(prefix="/api/scans", tags=["scans"])

repository = ScanRepository()
scanner = DocumentScanner()


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
        File(description="Photo of a document"),
    ],
    color_mode: Annotated[ColorMode, Form()] = "color",
    color_correction: Annotated[bool, Form()] = True,
    soften_colors: Annotated[float, Form(ge=0, le=1)] = 0.0,
) -> ScanOut:
    """Detecta, endereza, mejora y guarda un documento."""

    options = ScanOptions(
        color_mode=color_mode,
        color_correction=color_correction,
        soften_colors=soften_colors,
    )

    # Leer y validar la fotografía.
    upload = await read_upload(file)

    # Detectar las esquinas y corregir la perspectiva.
    # DocumentNotFound se traduce automáticamente a HTTP 422.
    processed_image, corners = scanner.scan(
        upload.photo.image
    )

    # Aplicar corrección de color si está habilitada.
    if color_correction:
        processed_image = ColorCorrectionFilter().apply(
            processed_image
        )

    # Aclarar colores intensos si el usuario lo solicita.
    if soften_colors > 0:
        processed_image = SoftenColorsFilter(
            soften_colors
        ).apply(processed_image)

    # Aplicar el modo de color solicitado.
    color_filters = {
        "color": Color(),
        "grayscale": GrayscaleFilter(),
        "bw": BWFilter(),
    }

    processed_image = color_filters[color_mode].apply(
        processed_image
    )

    # Convertir el resultado a PNG.
    image_png = encode_png(processed_image)

    # Generar identificador y nombre seguro.
    scan_id = uuid.uuid4().hex

    extension = upload.photo.format.lower()
    original_filename = f"{scan_id}-original.{extension}"

    # Preparar los metadatos.
    scan = ScanOut(
        id=scan_id,
        original_name=upload.name,
        url=f"/media/{scan_id}.png",
        original_url=f"/media/{original_filename}",
        width=processed_image.shape[1],
        height=processed_image.shape[0],
        size_bytes=len(image_png),
        corners=corners.tolist(),
        options=options,
        created_at=datetime.now(),
    )

    # El repositorio guarda PNG, original y JSON.
    return repository.save(
        scan=scan,
        image_png=image_png,
        original_content=upload.content,
    )


@router.get(
    "",
    response_model=list[ScanOut],
    summary="List the scans, newest first",
)
def list_scans() -> list[ScanOut]:
    """Lista los escaneos del más reciente al más antiguo."""
    return repository.list()


@router.get(
    "/{scan_id}",
    response_model=ScanOut,
    summary="Get a scan",
    responses={
        404: {"model": ErrorOut},
    },
)
def get_scan(scan_id: str) -> ScanOut:
    """Recupera los metadatos de un escaneo."""
    return repository.get(scan_id)


@router.delete(
    "/{scan_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a scan and its files",
    responses={
        404: {"model": ErrorOut},
    },
)
def delete_scan(scan_id: str) -> None:
    """Elimina el escaneo, la fotografía original y los metadatos."""
    repository.delete(scan_id)

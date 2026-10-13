from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.config import MEDIA_URL
from app.errors import ScanNotFound
from app.schemas import ColorMode, ErrorOut, ScanOptions, ScanOut
from app.storage import ScanRepository
from app.uploads import read_upload
from docscan.imageio import encode_png
from docscan.scanner import Scanner

router = APIRouter(prefix="/api/scans", tags=["scans"])
repo = ScanRepository()


@router.post(
    "",
    response_model=ScanOut,
    status_code=status.HTTP_201_CREATED,
    summary="Scan a photo and save the result",
    responses={400: {"model": ErrorOut}, 413: {"model": ErrorOut}, 422: {"model": ErrorOut}},
)
async def create_scan(
    file: Annotated[UploadFile, File(description="Photo of a document")],
    color_mode: Annotated[ColorMode, Form()] = "color",
    color_correction: Annotated[bool, Form()] = True,
    soften_colors: Annotated[float, Form(ge=0, le=1)] = 0.0,
):
    options = ScanOptions(color_mode=color_mode, color_correction=color_correction, soften_colors=soften_colors)
    upload = await read_upload(file)

    scanner = Scanner()
    result = scanner.scan(
        upload.photo.image,
        color_mode=color_mode,
        color_correction=color_correction,
        soften_colors=soften_colors,
    )

    extension_map = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }
    original_ext = extension_map.get(upload.photo.format.upper(), ".jpg")

    scan_id = repo.save(
        original_name=upload.name,
        original_bytes=upload.content,
        original_ext=original_ext,
        scan_bytes=encode_png(result.image),
        width=result.image.shape[1],
        height=result.image.shape[0],
        corners=[[float(x), float(y)] for x, y in result.corners],
        options=options.model_dump(),
    )

    return repo.get(scan_id)


@router.get("", response_model=list[ScanOut], summary="List the scans, newest first")
def list_scans():
    return repo.list()


@router.get("/{scan_id}", response_model=ScanOut, summary="Get a scan", responses={404: {"model": ErrorOut}})
def get_scan(scan_id: str):
    return repo.get(scan_id)


@router.delete(
    "/{scan_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a scan and its files",
    responses={404: {"model": ErrorOut}},
)
def delete_scan(scan_id: str):
    repo.delete(scan_id)
    return None
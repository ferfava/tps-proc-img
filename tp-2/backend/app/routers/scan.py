from docscan.filters import SoftenColorsFilter
from PIL import ImageColor
from docscan.filters import GrayscaleFilter, BWFilter, Color, ColorCorrectionFilter
from docscan.imageio import encode_png
from datetime import datetime
from typing import Annotated
import uuid
from app.config import STORAGE_DIR
from  docscan.detect import detect_document

from fastapi import APIRouter, File, Form, UploadFile, status

from app.schemas import ColorMode, ErrorOut, ScanOptions, ScanOut
from app.uploads import read_upload

router = APIRouter(prefix="/api/scans", tags=["scans"])


def empty_scan(scan_id: str = "", original_name: str = "", options: ScanOptions | None = None) -> ScanOut:
    """Placeholder answer with the right shape. Delete it once the endpoints are implemented."""
    return ScanOut(
        id=scan_id, original_name=original_name, url="", original_url="",
        width=0, height=0, size_bytes=0, corners=[],
        options=options or ScanOptions(), created_at=datetime.now(),
    )


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
    upload = await read_upload(file)  # given: 413 FILE_TOO_LARGE / 400 INVALID_FILE / 400 UNSUPPORTED_FORMAT
    # TODO (team): scan upload.photo.image with docscan, save the result (PNG), the original
    # photo (upload.content) and the metadata, and return the saved scan.
    

    COLOR_FILTERS = {
        "grayscale": GrayscaleFilter(),
        "bw": BWFilter(),
        "color": Color()
    }

    image = upload.photo.image

    
    

    if color_correction :
        new_image = ColorCorrectionFilter().apply(image)
    else:
        new_image = image

    new_image = SoftenColorsFilter(soften_colors).apply(new_image)

    filter = COLOR_FILTERS.get(color_mode)
    
    if filter:
        new_image = filter.apply(new_image)   
    else:
        new_image = new_image

    image_png = encode_png(new_image)
    
    scan_id = uuid.uuid4().hex
    (STORAGE_DIR / f"{scan_id}.png").write_bytes(image_png)
    extension = upload.photo.format.lower()
    (STORAGE_DIR / f"{scan_id}-original.{extension}").write_bytes(upload.content)

    scan = ScanOut(
        id=scan_id,
        original_name=upload.name,
        url=f"/media/{scan_id}.png",
        original_url=f"/media/{scan_id}-original.{extension}",
        width=new_image.shape[1],
        height=new_image.shape[0],
        size_bytes=len(image_png),
        corners=[],
        options=options,
        created_at=datetime.now(),
    )

    (STORAGE_DIR / f"{scan_id}.json").write_text(scan.model_dump_json())
   
    return scan

@router.get("", response_model=list[ScanOut], summary="List the scans, newest first")
def list_scans():
    # TODO (team): read the saved scans.
    return []


@router.get("/{scan_id}", response_model=ScanOut, summary="Get a scan", responses={404: {"model": ErrorOut}})
def get_scan(scan_id: str):
    # TODO (team): read the scan, or answer 404 SCAN_NOT_FOUND.
    return empty_scan(scan_id)


@router.delete(
    "/{scan_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a scan and its files",
    responses={404: {"model": ErrorOut}},
)
def delete_scan(scan_id: str):
    # TODO (team): delete the files of the scan, or answer 404 SCAN_NOT_FOUND.
    return None

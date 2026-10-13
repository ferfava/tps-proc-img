import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import STORAGE_DIR
from app.errors import ScanNotFound
from app.schemas import ScanOut


class ScanRepository:
    """Gestiona el almacenamiento y la metadata de los escaneos."""

    def __init__(self, base_dir: str | Path = STORAGE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_scan_id(self, scan_id: str) -> str:
        if not scan_id or not re.fullmatch(r"[0-9a-fA-F]{32}", scan_id):
            raise ValueError(f"ID inválido: {scan_id!r}")
        return scan_id

    def _scan_path(self, scan_id: str, suffix: str = "") -> Path:
        scan_id = self._validate_scan_id(scan_id)
        return self.base_dir / f"{scan_id}{suffix}"

    def save(
        self,
        *,
        original_name: str,
        original_bytes: bytes,
        original_ext: str,
        scan_bytes: bytes,
        width: int,
        height: int,
        corners: list[list[float]],
        options: dict[str, Any],
    ) -> str:
        scan_id = uuid.uuid4().hex

        scan_path = self._scan_path(scan_id, ".png")
        scan_path.write_bytes(scan_bytes)

        original_path = self._scan_path(scan_id, f"-original{original_ext}")
        original_path.write_bytes(original_bytes)

        payload = {
            "id": scan_id,
            "original_name": original_name,
            "url": f"/media/{scan_id}.png",
            "original_url": f"/media/{scan_id}-original{original_ext}",
            "width": width,
            "height": height,
            "size_bytes": len(scan_bytes),
            "corners": corners,
            "options": options,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

        metadata_path = self._scan_path(scan_id, ".json")
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False))

        return scan_id

    def get(self, scan_id: str) -> ScanOut:
        scan_id = self._validate_scan_id(scan_id)
        metadata_path = self._scan_path(scan_id, ".json")
        if not metadata_path.exists():
            raise ScanNotFound(scan_id)

        payload = json.loads(metadata_path.read_text())
        return ScanOut(**payload)

    def list(self) -> list[ScanOut]:
        scans: list[ScanOut] = []
        for json_path in sorted(self.base_dir.glob("*.json")):
            if json_path.name.endswith("-original.json"):
                continue
            try:
                payload = json.loads(json_path.read_text())
                scans.append(ScanOut(**payload))
            except Exception:
                continue

        scans.sort(key=lambda s: s.created_at, reverse=True)
        return scans

    def delete(self, scan_id: str) -> None:
        scan_id = self._validate_scan_id(scan_id)
        png = self._scan_path(scan_id, ".png")
        json_file = self._scan_path(scan_id, ".json")
        candidates = list(self.base_dir.glob(f"{scan_id}-original*"))

        if not png.exists() and not json_file.exists() and not candidates:
            raise ScanNotFound(scan_id)

        for path in [png, json_file, *candidates]:
            if path.exists():
                path.unlink()

    def __len__(self) -> int:
        return len(list(self.base_dir.glob("*.json")))
from app.errors import ScanNotFound
from pathlib import Path

from app.config import STORAGE_DIR
from app.schemas import ScanOut


class ScanRepository:
    """Administra los escaneos almacenados en disco."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or STORAGE_DIR

    
    def save(
        self,
        scan: ScanOut,
        image_png: bytes,
        original_content: bytes,
    ) -> ScanOut:
        """Guarda el escaneo, la foto original y los metadatos."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        scan_id = scan.id

        # Aceptar únicamente identificadores hexadecimales.
        if len(scan_id) != 32 or any(
            char not in "0123456789abcdef" for char in scan_id
        ):
            raise ValueError("Identificador de escaneo inválido")

        # Obtener la extensión del original desde los metadatos.
        original_name = Path(scan.original_url).name

        # Verificar que corresponda a este escaneo.
        if not original_name.startswith(f"{scan_id}-original."):
            raise ValueError("Nombre de archivo original inválido")

        png_path = self.storage_dir / f"{scan_id}.png"
        original_path = self.storage_dir / original_name
        metadata_path = self.storage_dir / f"{scan_id}.json"

        # Guardar las imágenes.
        png_path.write_bytes(image_png)
        original_path.write_bytes(original_content)

        # Guardar los metadatos.
        metadata_path.write_text(
            scan.model_dump_json(),
            encoding="utf-8",
        )

        return scan


    
    def get(self, scan_id: str) -> ScanOut:
        """Recupera los metadatos de un escaneo guardado."""

        # Validar el identificador antes de construir una ruta.
        if len(scan_id) != 32 or any(
            char not in "0123456789abcdef" for char in scan_id
        ):
            raise ScanNotFound(scan_id)

        # Construir la ruta del archivo JSON.
        metadata_path = self.storage_dir / f"{scan_id}.json"

        # Verificar que el archivo exista.
        if not metadata_path.is_file():
            raise ScanNotFound(scan_id)

        # Leer y validar los metadatos.
        content = metadata_path.read_text(encoding="utf-8")

        return ScanOut.model_validate_json(content)


    
    def list(self) -> list[ScanOut]:
        """Recupera todos los escaneos, del más reciente al más antiguo."""

        scans = []

        for metadata_path in self.storage_dir.glob("*.json"):
            content = metadata_path.read_text(encoding="utf-8")

            scan = ScanOut.model_validate_json(content)
            scans.append(scan)

        scans.sort(
            key=lambda scan: scan.created_at,
            reverse=True
        )

        return scans


    
    def delete(self, scan_id: str) -> None:
        """Elimina un escaneo y todos sus archivos asociados."""

        # Recuperar el escaneo y validar su identificador.
        scan = self.get(scan_id)

        # Obtener únicamente los nombres de los archivos.
        # Esto evita utilizar rutas externas al almacenamiento.
        filenames = [
            f"{scan_id}.png",
            Path(scan.original_url).name,
            f"{scan_id}.json",
        ]

        # Eliminar los archivos asociados.
        for filename in filenames:
            file_path = self.storage_dir / filename

            if file_path.is_file():
                file_path.unlink()


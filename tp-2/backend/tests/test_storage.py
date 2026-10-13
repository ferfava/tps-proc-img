"""
Tests para verificar el almacenamiento, recuperación, listado y seguridad
del repositorio de escaneos (ScanRepository).
"""

from pathlib import Path

import pytest

from app.errors import ScanNotFound
from app.storage import ScanRepository


def save_fake_scan(repo: ScanRepository) -> str:
    """Guarda un escaneo de mentira y devuelve su ID generado."""
    return repo.save(
        original_name="factura.jpg",
        original_bytes=b"foto",
        original_ext=".jpg",
        scan_bytes=b"escaneo",
        width=10,
        height=20,
        corners=[[0, 0], [10, 0], [10, 20], [0, 20]],
        options={
            "color_mode": "color",
            "color_correction": True,
            "soften_colors": 0.0
        },
    )


def test_save_creates_three_files(tmp_path: Path) -> None:
    """Prueba que guardar un escaneo cree exactamente 3 archivos (PNG, JSON y original)."""
    repo = ScanRepository(tmp_path)
    save_fake_scan(repo)
    
    # Comprobamos que existan exactamente 3 elementos en la carpeta temporal
    assert len(list(tmp_path.iterdir())) == 3


def test_get_returns_saved_scan(tmp_path: Path) -> None:
    """Prueba que recuperar un escaneo mediante su ID devuelva los datos correctos guardados."""
    repo = ScanRepository(tmp_path)
    scan_id = save_fake_scan(repo)
    
    scan = repo.get(scan_id)
    assert scan.original_name == "factura.jpg"
    assert scan.width == 10
    assert scan.height == 20


def test_list_returns_all_scans(tmp_path: Path) -> None:
    """Prueba que al listar los escaneos se devuelvan todos los almacenados."""
    repo = ScanRepository(tmp_path)
    save_fake_scan(repo)
    save_fake_scan(repo)
    
    scans = repo.list()
    assert len(scans) == 2


def test_delete_removes_all_files(tmp_path: Path) -> None:
    """Prueba que borrar un escaneo elimine por completo todos sus archivos asociados."""
    repo = ScanRepository(tmp_path)
    scan_id = save_fake_scan(repo)
    
    repo.delete(scan_id)
    
    # La carpeta temporal debe quedar vacía
    assert list(tmp_path.iterdir()) == []


def test_get_unknown_id_raises_scan_not_found(tmp_path: Path) -> None:
    """Prueba que buscar un ID válido pero inexistente lance la excepción ScanNotFound."""
    repo = ScanRepository(tmp_path)
    
    # 32 ceros (un identificador con formato de hash válido pero que no existe)
    unknown_id = "0" * 32
    
    with pytest.raises(ScanNotFound):
        repo.get(unknown_id)


def test_invalid_id_is_rejected(tmp_path: Path) -> None:
    """Prueba que se rechace un ID malicioso o inválido (intento de path traversal) lanzando ScanNotFound."""
    repo = ScanRepository(tmp_path)
    
    # Intento de salir de la carpeta mediante ../config
    malicious_id = "../config"
    
    with pytest.raises(ScanNotFound):
        repo.get(malicious_id)
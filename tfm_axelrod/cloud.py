"""Subida de la salida del simulador a Azure Data Lake Storage Gen2.

Usa el SDK oficial (``azure-storage-file-datalake``), que es una dependencia
opcional: ``uv sync --extra azure``. Sin él, el simulador funciona igual en
local; solo falla (con un mensaje claro) si se pide ``--upload-prefix``.

La cadena de conexión se lee de la variable de entorno
``AZURE_STORAGE_CONNECTION_STRING`` y nunca se imprime.
"""

from __future__ import annotations

import os
from pathlib import Path

CONN_ENV = "AZURE_STORAGE_CONNECTION_STRING"


class UploadError(RuntimeError):
    """Error de configuración o de subida al data lake."""


def get_filesystem_client(filesystem: str, connection_string: str | None = None):
    """Cliente del sistema de ficheros ADLS Gen2 (no hace llamadas de red)."""
    conn = connection_string or os.environ.get(CONN_ENV)
    if not conn:
        raise UploadError(f"Falta la variable de entorno {CONN_ENV}")
    try:
        from azure.storage.filedatalake import DataLakeServiceClient
    except ImportError as exc:
        raise UploadError("Falta el SDK de Azure: instala el extra con "
                          "`uv sync --extra azure` (paquete azure-storage-file-datalake)") from exc
    try:
        service = DataLakeServiceClient.from_connection_string(conn)
    except Exception as exc:  # el mensaje del SDK no incluye la clave
        raise UploadError(f"Cadena de conexión no válida ({type(exc).__name__})") from exc
    return service.get_file_system_client(filesystem)


def check_filesystem(fs_client) -> None:
    """Comprueba (con una llamada de red) que el sistema de ficheros existe."""
    try:
        exists = fs_client.exists()
    except Exception as exc:
        raise UploadError(f"No se pudo acceder al data lake: {type(exc).__name__}: {exc}") from exc
    if not exists:
        raise UploadError(f"El sistema de ficheros {fs_client.file_system_name!r} no existe")


def remote_path(prefix: str, local_root: Path, file: Path) -> str:
    """Ruta en el lake: ``<prefix>/<ruta relativa a local_root>`` con '/'."""
    rel = Path(file).relative_to(local_root).as_posix()
    prefix = prefix.strip("/")
    return f"{prefix}/{rel}" if prefix else rel


def upload_files(files: list[Path], local_root: str | Path, prefix: str, fs_client) -> list[str]:
    """Sube ``files`` manteniendo su estructura relativa a ``local_root``.

    Sobrescribe si ya existen (la misma configuración genera el mismo
    nombre de fichero). Devuelve las rutas remotas subidas.
    """
    local_root = Path(local_root)
    uploaded = []
    for file in sorted(Path(f) for f in files):
        dest = remote_path(prefix, local_root, file)
        size = file.stat().st_size
        try:
            with open(file, "rb") as fh:
                fs_client.get_file_client(dest).upload_data(fh, length=size, overwrite=True)
        except Exception as exc:
            raise UploadError(f"Fallo al subir {dest}: {type(exc).__name__}: {exc}") from exc
        uploaded.append(dest)
        print(f"  subido {dest} ({size / 1e6:.2f} MB)", flush=True)
    return uploaded


def list_files(fs_client, prefix: str) -> list[str]:
    """Rutas de los ficheros (no carpetas) bajo ``prefix``, recursivamente."""
    prefix = prefix.strip("/")
    try:
        return sorted(p.name for p in fs_client.get_paths(path=prefix, recursive=True)
                      if not p.is_directory)
    except Exception as exc:
        raise UploadError(f"No se pudo listar {fs_client.file_system_name}/{prefix}: "
                          f"{type(exc).__name__}: {exc}") from exc


def download_prefix(fs_client, prefix: str, local_dir: str | Path) -> list[Path]:
    """Descarga todo lo que hay bajo ``prefix`` en ``local_dir``, manteniendo la estructura."""
    prefix = prefix.strip("/")
    local_dir = Path(local_dir)
    out = []
    for name in list_files(fs_client, prefix):
        dest = local_dir / name[len(prefix):].lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest, "wb") as fh:
                fs_client.get_file_client(name).download_file().readinto(fh)
        except Exception as exc:
            raise UploadError(f"Fallo al descargar {name}: {type(exc).__name__}: {exc}") from exc
        out.append(dest)
    return out


def delete_directory(fs_client, path: str) -> bool:
    """Borra la carpeta ``path`` del lake si existe. Devuelve si existía."""
    path = path.strip("/")
    if not path:
        raise UploadError("Se ha intentado borrar la raíz del sistema de ficheros")
    try:
        directory = fs_client.get_directory_client(path)
        if not directory.exists():
            return False
        directory.delete_directory()
    except Exception as exc:
        raise UploadError(f"No se pudo borrar {path}: {type(exc).__name__}: {exc}") from exc
    print(f"  borrada la versión anterior de {path}/", flush=True)
    return True

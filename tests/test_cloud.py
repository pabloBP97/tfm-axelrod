"""Lógica de subida al data lake con el cliente de Azure simulado (sin red)."""

import sys
from pathlib import Path

import pytest

from tfm_axelrod import cloud, simulate

FAKE_CONN = ("DefaultEndpointsProtocol=https;AccountName=fakeaccount;"
             "AccountKey=ZmFrZWtleQ==;EndpointSuffix=core.windows.net")


class FakeFileClient:
    def __init__(self, fs, path):
        self.fs, self.path = fs, path

    def upload_data(self, data, length=None, overwrite=False):
        if any(part in self.path for part in self.fs.fail_on):
            raise IOError("conexión cortada")
        content = data.read() if hasattr(data, "read") else data
        assert length == len(content)
        self.fs.files[self.path] = {"data": content, "overwrite": overwrite}


class FakeFileSystemClient:
    def __init__(self, exists=True, fail_on=()):
        self.file_system_name = "lake"
        self.files: dict[str, dict] = {}
        self._exists = exists
        self.fail_on = set(fail_on)

    def exists(self):
        return self._exists

    def get_file_client(self, path):
        return FakeFileClient(self, path)


@pytest.fixture
def fake_fs(monkeypatch):
    fs = FakeFileSystemClient()
    monkeypatch.setattr(cloud, "get_filesystem_client", lambda name, conn=None: fs)
    return fs


ARGS = ["--preset", "small", "--noise", "0", "0.05", "--turns", "10", "20",
        "--repetitions", "1", "--seed", "1"]


def test_remote_path():
    root = Path("out")
    f = root / "roundrobin" / "noise=0.0" / "turns=10" / "ro_x.parquet"
    assert cloud.remote_path("raw_cloud/", root, f) == "raw_cloud/roundrobin/noise=0.0/turns=10/ro_x.parquet"
    assert cloud.remote_path("", root, f) == "roundrobin/noise=0.0/turns=10/ro_x.parquet"


def test_upload_keeps_structure_and_content(tmp_path, fake_fs, capsys):
    simulate.main(ARGS + ["--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    local = sorted((tmp_path / "roundrobin").rglob("*.parquet"))
    assert len(local) == 4 and len(fake_fs.files) == 4
    for f in local:
        rel = f.relative_to(tmp_path).as_posix()
        remote = fake_fs.files[f"raw_cloud/{rel}"]
        assert remote["data"] == f.read_bytes() and remote["overwrite"] is True
    assert all(p.startswith("raw_cloud/roundrobin/noise=") for p in fake_fs.files)
    out = capsys.readouterr().out
    rows = 15 * 2 * (10 + 20) * 1 * 2
    assert f"RESUMEN filas={rows} ficheros_subidos=4" in out


def test_only_files_of_this_run_are_uploaded(tmp_path, fake_fs):
    other = tmp_path / "roundrobin" / "noise=0.0" / "turns=10" / "ro_otro.parquet"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"de otra ejecucion")
    simulate.main(ARGS + ["--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    assert len(fake_fs.files) == 4
    assert not any(p.endswith("ro_otro.parquet") for p in fake_fs.files)


def test_evolutionary_upload(tmp_path, fake_fs):
    simulate.main(["--mode", "evolutionary", "--preset", "small", "--noise", "0", "--turns", "10",
                   "--generations", "3", "--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    assert len(fake_fs.files) == 1
    assert next(iter(fake_fs.files)).startswith("raw_cloud/evolutionary/ev_")


def test_local_use_without_upload_does_not_touch_azure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no debería llamarse")
    monkeypatch.setattr(cloud, "get_filesystem_client", boom)
    simulate.main(ARGS + ["--out", str(tmp_path)])
    assert len(list(tmp_path.rglob("*.parquet"))) == 4


def test_missing_connection_string_fails_before_simulating(tmp_path, monkeypatch):
    monkeypatch.delenv(cloud.CONN_ENV, raising=False)
    with pytest.raises(SystemExit) as exc:
        simulate.main(ARGS + ["--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    assert cloud.CONN_ENV in str(exc.value) and exc.value.code != 0
    assert not (tmp_path / "roundrobin").exists()


def test_missing_sdk_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv(cloud.CONN_ENV, FAKE_CONN)
    monkeypatch.setitem(sys.modules, "azure.storage.filedatalake", None)  # import falla
    with pytest.raises(SystemExit) as exc:
        simulate.main(ARGS + ["--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    assert "SDK de Azure" in str(exc.value)
    assert not (tmp_path / "roundrobin").exists()


def test_missing_filesystem_fails_before_simulating(tmp_path, monkeypatch):
    fs = FakeFileSystemClient(exists=False)
    monkeypatch.setattr(cloud, "get_filesystem_client", lambda name, conn=None: fs)
    with pytest.raises(SystemExit) as exc:
        simulate.main(ARGS + ["--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    assert "no existe" in str(exc.value)
    assert not (tmp_path / "roundrobin").exists()


def test_upload_error_exits_nonzero(tmp_path, monkeypatch, capsys):
    fs = FakeFileSystemClient(fail_on={"noise=0.05/turns=10/"})  # falla a mitad de la subida
    monkeypatch.setattr(cloud, "get_filesystem_client", lambda name, conn=None: fs)
    with pytest.raises(SystemExit) as exc:
        simulate.main(ARGS + ["--out", str(tmp_path), "--upload-prefix", "raw_cloud"])
    assert "Fallo al subir" in str(exc.value) and "conexión cortada" in str(exc.value)
    assert "RESUMEN" not in capsys.readouterr().out


def test_cli_exit_code_is_nonzero(tmp_path):
    """El proceso real termina con código distinto de 0 si falta la configuración."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != cloud.CONN_ENV}
    r = subprocess.run([sys.executable, "-m", "tfm_axelrod.simulate", *ARGS, "--out",
                        str(tmp_path), "--upload-prefix", "raw_cloud"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and cloud.CONN_ENV in r.stderr


class _P:
    def __init__(self, name, is_directory=False):
        self.name, self.is_directory = name, is_directory


class FakeLakeForDownload:
    file_system_name = "lake"

    def __init__(self, files, fail_listing=False):
        self.files, self.fail_listing, self.deleted = dict(files), fail_listing, []

    def get_paths(self, path, recursive=True):
        if self.fail_listing:
            raise IOError("ResourceNotFound")
        return [_P(path, True)] + [_P(n) for n in self.files if n.startswith(path + "/")]

    def get_file_client(self, path):
        data = self.files[path]

        class _Dl:
            def readinto(self, fh):
                fh.write(data)

        class _F:
            def download_file(self):
                return _Dl()
        return _F()

    def get_directory_client(self, path):
        lake = self

        class _D:
            def exists(self):
                return any(n.startswith(path + "/") for n in lake.files)

            def delete_directory(self):
                lake.deleted.append(path)
        return _D()


def test_download_prefix_keeps_structure_and_skips_other_prefixes(tmp_path):
    lake = FakeLakeForDownload({
        "raw/roundrobin/a.parquet": b"a", "raw/roundrobin/sub/b.parquet": b"bb",
        "raw_cloud/roundrobin/a.parquet": b"no", "raw/evolutionary/e.parquet": b"e"})
    files = cloud.download_prefix(lake, "raw/roundrobin/", tmp_path / "rr")
    assert sorted(f.relative_to(tmp_path).as_posix() for f in files) == ["rr/a.parquet", "rr/sub/b.parquet"]
    assert (tmp_path / "rr" / "sub" / "b.parquet").read_bytes() == b"bb"


def test_download_and_delete_errors(tmp_path):
    with pytest.raises(cloud.UploadError, match="No se pudo listar"):
        cloud.download_prefix(FakeLakeForDownload({}, fail_listing=True), "raw/roundrobin", tmp_path)
    lake = FakeLakeForDownload({"curated/t/x.parquet": b"x"})
    assert cloud.delete_directory(lake, "curated/t") and lake.deleted == ["curated/t"]
    assert not cloud.delete_directory(lake, "curated/otra")
    with pytest.raises(cloud.UploadError, match="raíz"):
        cloud.delete_directory(lake, "/")


def test_real_sdk_client_builds_offline():
    pytest.importorskip("azure.storage.filedatalake")
    fs = cloud.get_filesystem_client("lake", FAKE_CONN)
    assert fs.file_system_name == "lake" and "fakeaccount" in fs.url

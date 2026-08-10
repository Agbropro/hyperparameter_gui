from pathlib import Path

import pytest

from infrastructure.configuration import ServerSettings, load_server_settings


def test_load_server_settings(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  host: 0.0.0.0\n  port: 9010\n  reload: false\n", encoding="utf-8")
    assert load_server_settings(path) == ServerSettings(host="0.0.0.0", port=9010, reload=False)


@pytest.mark.parametrize("port", [0, 65536, "eight-thousand", True])
def test_rejects_invalid_port(tmp_path: Path, port):
    path = tmp_path / "config.yaml"
    path.write_text(f"server:\n  host: 127.0.0.1\n  port: {str(port).lower()}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="server.port"):
        load_server_settings(path)

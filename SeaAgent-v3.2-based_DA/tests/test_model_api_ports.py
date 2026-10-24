from copy import deepcopy
from urllib.parse import urlsplit

import pytest
import yaml

from config import load_config, settings


def test_public_settings_exposes_current_vlm_and_embedding_ports():
    config = load_config()
    public = settings.public_settings(config)["settings"]

    assert public["llm"]["api_port"] == urlsplit(config["llm"]["base_url"]).port
    assert public["embedding"]["api_port"] == urlsplit(config["embedding"]["base_url"]).port


def test_updating_api_ports_preserves_service_hosts_and_paths(tmp_path, monkeypatch):
    runtime_file = tmp_path / "runtime.yaml"
    monkeypatch.setattr(settings, "RUNTIME_FILE", runtime_file)
    config = load_config()

    result = settings.update_settings(
        config,
        {"settings": {"llm": {"api_port": 9000}, "embedding": {"api_port": 9001}}},
    )

    assert config["llm"]["base_url"] == "http://localhost:9000/v1"
    assert config["embedding"]["base_url"] == "http://localhost:9001/v1"
    assert result["settings"]["llm"]["api_port"] == 9000
    assert result["settings"]["embedding"]["api_port"] == 9001
    assert yaml.safe_load(runtime_file.read_text(encoding="utf-8")) == {
        "llm": {"base_url": "http://localhost:9000/v1"},
        "embedding": {"base_url": "http://localhost:9001/v1"},
    }


@pytest.mark.parametrize("port", [0, 65536, 8000.5, True, "not-a-port"])
def test_api_port_validation_rejects_invalid_values(tmp_path, monkeypatch, port):
    runtime_file = tmp_path / "runtime.yaml"
    monkeypatch.setattr(settings, "RUNTIME_FILE", runtime_file)
    config = load_config()
    original_url = config["llm"]["base_url"]

    with pytest.raises(ValueError):
        settings.update_settings(config, {"settings": {"llm": {"api_port": port}}})

    assert config["llm"]["base_url"] == original_url
    assert not runtime_file.exists()


def test_port_update_preserves_ipv6_authority_and_url_path():
    config = deepcopy(load_config())
    config["llm"]["base_url"] = "https://user:secret@[::1]:7890/openai/v1?tenant=qa"

    result = settings._url_with_api_port(config, "llm.api_port", 8123)

    assert result == "https://user:secret@[::1]:8123/openai/v1?tenant=qa"


def test_reset_restores_default_service_ports(tmp_path, monkeypatch):
    runtime_file = tmp_path / "runtime.yaml"
    runtime_file.write_text("llm:\n  base_url: http://localhost:9000/v1\n", encoding="utf-8")
    monkeypatch.setattr(settings, "RUNTIME_FILE", runtime_file)
    defaults = load_config()
    monkeypatch.setattr(settings, "load_config", lambda: defaults)
    config = deepcopy(defaults)
    config["llm"]["base_url"] = "http://localhost:9000/v1"

    result = settings.reset_settings(config)

    assert not runtime_file.exists()
    assert config["llm"]["base_url"] == defaults["llm"]["base_url"]
    assert result["settings"]["llm"]["api_port"] == urlsplit(defaults["llm"]["base_url"]).port

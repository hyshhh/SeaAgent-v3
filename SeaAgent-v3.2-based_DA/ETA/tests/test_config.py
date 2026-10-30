from pathlib import Path

from eta.config import ETA_ROOT, load_config, resolve_source_path


def test_default_config_and_smoke_merge():
    default = load_config()
    smoke = load_config("configs/smoke.yaml")
    assert default["data"]["history_pool_size"] == 3
    assert smoke["data"]["history_pool_size"] == 3
    assert smoke["train"]["epochs"] == 2
    assert Path(smoke["paths"]["processed_dir"]).is_absolute()
    assert Path(smoke["paths"]["source_data_root"]).is_absolute()


def test_source_path_resolves_relative_to_application():
    config = load_config()
    path = resolve_source_path(config, config["data"]["tracks_csv"])
    assert path == Path(config["paths"]["source_data_root"]) / "data/memory/tracks.csv"
    assert ETA_ROOT.name == "ETA"

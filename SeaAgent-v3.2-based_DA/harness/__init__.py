"""Public harness API with lazy imports for lightweight state/config tooling."""

__all__ = ["SeaVideoHarness", "build_middleware", "build_model", "build_tools", "run_harness"]


def __getattr__(name: str):
    if name == "SeaVideoHarness" or name == "run_harness":
        from .runtime import SeaVideoHarness, run_harness
        return SeaVideoHarness if name == "SeaVideoHarness" else run_harness
    if name == "build_middleware":
        from .middleware import build_middleware
        return build_middleware
    if name == "build_model":
        from .model import build_model
        return build_model
    if name == "build_tools":
        from .tools import build_tools
        return build_tools
    raise AttributeError(name)

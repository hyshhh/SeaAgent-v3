from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from harness.tool_guard import ToolConcurrencyGate


def _run(gate: ToolConcurrencyGate, tool_name: str) -> tuple[int, int]:
    with gate.slot(tool_name):
        with gate._condition:
            if tool_name == "verify_vessel_images":
                active = gate._active_vlm
            else:
                active = gate._active_other
        time.sleep(0.02)
        return active, 0


def test_vlm_calls_are_limited_to_five_concurrent_executions():
    gate = ToolConcurrencyGate(vlm_limit=5)
    with ThreadPoolExecutor(max_workers=10) as pool:
        active_counts = list(
            pool.map(lambda _: _run(gate, "verify_vessel_images")[0], range(10))
        )

    assert max(active_counts) == 5
    assert gate._active_vlm == 0
    assert gate._active_other == 0


def test_non_vlm_calls_remain_serial():
    gate = ToolConcurrencyGate(vlm_limit=5)
    with ThreadPoolExecutor(max_workers=10) as pool:
        active_counts = list(
            pool.map(lambda _: _run(gate, "recognize_tracks")[0], range(10))
        )

    assert max(active_counts) == 1
    assert gate._active_vlm == 0
    assert gate._active_other == 0


def test_non_vlm_calls_wait_while_vlm_calls_are_running():
    gate = ToolConcurrencyGate(vlm_limit=5)
    started = threading.Event()
    release = threading.Event()
    observed: list[str] = []

    def run_vlm() -> None:
        with gate.slot("verify_vessel_images"):
            started.set()
            release.wait(2)
            observed.append("vlm")

    def run_other() -> None:
        started.wait(2)
        with gate.slot("recognize_tracks"):
            observed.append("other")

    with ThreadPoolExecutor(max_workers=2) as pool:
        vlm_future = pool.submit(run_vlm)
        other_future = pool.submit(run_other)
        started.wait(2)
        time.sleep(0.05)
        assert other_future.done() is False
        release.set()
        vlm_future.result(timeout=2)
        other_future.result(timeout=2)

    assert observed == ["vlm", "other"]


def test_runtimes_share_the_process_wide_vlm_limit_gate():
    from harness.runtime import _shared_tool_concurrency_gate

    assert _shared_tool_concurrency_gate(5) is _shared_tool_concurrency_gate(5)

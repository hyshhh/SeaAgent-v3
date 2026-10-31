from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from harness.tool_guard import ToolConcurrencyGate


def _run(gate: ToolConcurrencyGate, tool_name: str) -> tuple[int, int]:
    with gate.slot(tool_name):
        with gate._condition:
            active = (
                gate._active_concurrent
                if tool_name in {"verify_vessel_images", "read_file", "read_shared_state"}
                else gate._active_exclusive
            )
        time.sleep(0.02)
        return active, 0


@pytest.mark.parametrize(
    "tool_names",
    [
        ("verify_vessel_images",) * 20,
        ("read_file",) * 20,
        ("read_shared_state",) * 20,
        (
            "verify_vessel_images",
            "read_file",
            "read_shared_state",
        )
        * 7,
    ],
)
def test_concurrent_read_tools_share_a_fifteen_slot_pool(tool_names):
    gate = ToolConcurrencyGate(concurrency_limit=15)
    with ThreadPoolExecutor(max_workers=20) as pool:
        active_counts = list(
            pool.map(lambda name: _run(gate, name)[0], tool_names)
        )

    assert max(active_counts) == 15
    assert gate._active_concurrent == 0
    assert gate._active_exclusive == 0


@pytest.mark.parametrize("tool_name", ["recognize_tracks", "update_execution_state"])
def test_non_concurrent_calls_remain_serial(tool_name):
    gate = ToolConcurrencyGate(concurrency_limit=15)
    with ThreadPoolExecutor(max_workers=10) as pool:
        active_counts = list(
            pool.map(lambda _: _run(gate, tool_name)[0], range(10))
        )

    assert max(active_counts) == 1
    assert gate._active_concurrent == 0
    assert gate._active_exclusive == 0


@pytest.mark.parametrize("read_tool", ["verify_vessel_images", "read_file", "read_shared_state"])
def test_non_concurrent_calls_wait_while_read_calls_are_running(read_tool):
    gate = ToolConcurrencyGate(concurrency_limit=15)
    started = threading.Event()
    release = threading.Event()
    observed: list[str] = []

    def run_read() -> None:
        with gate.slot(read_tool):
            started.set()
            release.wait(2)
            observed.append("read")

    def run_other() -> None:
        started.wait(2)
        with gate.slot("update_execution_state"):
            observed.append("other")

    with ThreadPoolExecutor(max_workers=2) as pool:
        read_future = pool.submit(run_read)
        other_future = pool.submit(run_other)
        started.wait(2)
        time.sleep(0.05)
        assert other_future.done() is False
        release.set()
        read_future.result(timeout=2)
        other_future.result(timeout=2)

    assert observed == ["read", "other"]


def test_runtimes_share_the_process_wide_concurrency_gate():
    from harness.runtime import _shared_tool_concurrency_gate

    assert _shared_tool_concurrency_gate(15) is _shared_tool_concurrency_gate(15)

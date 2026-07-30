"""
Utilities for cleaning up multiprocessing children on interrupted runs.

This is intentionally small and dependency-light. psutil is used when present
because it can find nested children reliably; the stdlib fallback still cleans
workers known to a ProcessPoolExecutor.
"""

from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from typing import Iterable

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal environments
    psutil = None


class ProcessPoolInterrupted(KeyboardInterrupt):
    """Raised when SIGINT/SIGTERM interrupts a managed process pool."""


def _executor_worker_pids(executor) -> set[int]:
    processes = getattr(executor, "_processes", None) or {}
    pids = set()
    for process in processes.values():
        pid = getattr(process, "pid", None)
        if pid:
            pids.add(pid)
    return pids


def _child_pids() -> set[int]:
    if psutil is None:
        return set()
    try:
        proc = psutil.Process(os.getpid())
        return {child.pid for child in proc.children(recursive=True)}
    except Exception:
        return set()


def _live_pids(pids: Iterable[int]) -> set[int]:
    live = set()
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        live.add(pid)
    return live


def _signal_pids(pids: Iterable[int], signum: int) -> None:
    for pid in sorted(_live_pids(pids), reverse=True):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"Warning: could not signal child process {pid}: {exc}", flush=True)


def cleanup_process_pool(executor, label: str = "process pool", grace_seconds: float = 5.0) -> None:
    """Cancel executor work and terminate currently known child processes."""
    worker_pids = _executor_worker_pids(executor)
    child_pids = _child_pids()
    target_pids = worker_pids | child_pids

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)
    except Exception as exc:
        print(f"Warning: failed to shutdown {label}: {exc}", flush=True)

    target_pids |= worker_pids | child_pids | _child_pids()
    target_pids = _live_pids(target_pids)
    if not target_pids:
        return

    print(f"Cleaning up {len(target_pids)} child process(es) from {label}...", flush=True)
    _signal_pids(target_pids, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        remaining = _live_pids(target_pids)
        if not remaining:
            return
        time.sleep(0.2)

    remaining = _live_pids(target_pids)
    if remaining:
        print(f"Force-killing {len(remaining)} child process(es) from {label}.", flush=True)
        _signal_pids(remaining, signal.SIGKILL)


@contextmanager
def managed_process_pool(executor, label: str = "process pool", grace_seconds: float = 5.0):
    """Ensure a ProcessPoolExecutor's children are cleaned on interrupts/errors."""
    old_handlers = {}

    def _raise_interrupted(signum, frame):
        raise ProcessPoolInterrupted(f"{label} interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_interrupted)
        except ValueError:
            pass

    try:
        yield executor
    except BaseException:
        cleanup_process_pool(executor, label=label, grace_seconds=grace_seconds)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

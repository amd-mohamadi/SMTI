"""
Parallel execution driver for SMTI event batches.

This module owns the *machinery* of running many events -- JAX device pinning,
the persistent fp32 batched-GPU worker, shape bucketing, the loader pool and the
post-processing pool -- and nothing about any particular project.  Everything
project-specific (how an event's inputs are loaded and filtered, what a finished
event writes to disk, which catalogue orders the queue) is supplied by the
caller as picklable, module-level callables.

Two entry points:

``run_event_batch_gpu``
    One persistent fp32 GPU worker samples shape-grouped batches while the
    driver post-processes the previous submission.  See the section header
    above ``_gpu_worker_main`` for the process layout.
``run_event_batch_cpu``
    Serial (``event_workers == 1``) or bounded CPU worker pool over prepared
    task tuples.

Caller-supplied callables
-------------------------
``make_load_task(index, total, dat_path) -> tuple``
    Build the picklable task tuple for one event's input loading.
``load_task_fn(task) -> prep dict``
    Module-level (spawn-picklable) function executed in the loader pool.
``make_post_task(prep, result_obj, reason) -> tuple``
    Build the picklable task tuple for one event's post-processing.  ``prep``
    arrives already shallow-copied; ``result_obj`` is ``None`` when the batched
    sampler produced no result and the event must fall back.
``post_task_fn(task) -> report dict``
    Module-level function executed in the post-processing pool.  Must return
    ``{"outcome": <result dict>, "batched": int, "fallback": int,
    "fallback_wall": float}``.

The prep dict returned by ``load_task_fn``
------------------------------------------
``kind``
    ``'ready'``, ``'skipped'`` or ``'failed'``.
``kind == 'ready'``
    ``event_id``, ``dat_path``, ``data`` (the event DataDict handed to the
    sampler), ``log_path``, ``index``, ``shape_key`` and ``bucket_key``.  The
    two keys are ``(n_pol, n_ar)`` tuples -- ``bucket_key`` padded (see
    ``round_up`` / ``batch_shape_key``) -- and decide which events share one
    compiled XLA program.
``kind == 'skipped'``
    ``dat_path`` and ``result``: the per-event result dict recorded as-is.
``kind == 'failed'``
    ``event_id``, ``dat_path``, ``error``, ``traceback``, ``log_path``.

The per-event ``outcome`` dicts these produce are accumulated into the list both
entry points return; the driver only reads ``event_id``, ``dat_path``, ``ok``,
``status``, ``path`` and ``log_path`` off them.
"""

from __future__ import annotations

import atexit
import gc
import multiprocessing as mp
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from .bounded_process_pool import bounded_process_pool_map
from .inversion_blackjax import InversionBlackJAX as Inversion

#: the three jitted programs built per batched configuration (batched:712-764)
GPU_PROGRAM_NAMES = ("init_fn", "step_fn", "read_back")

#: matches run_batched's per-stage progress line, e.g.
#: "[bucket 1/1 ...] stage   7: lambda min=0.04 ... finished=0/2 ESS ..."
_STAGE_LINE_RE = re.compile(
    r"stage\s+(?P<stage>\d+):.*finished=(?P<done>\d+)/(?P<total>\d+)"
)


def line_buffer_stdout() -> None:
    """Line-buffer stdout/stderr.

    The driver and the spawned worker share one stdout. When that is a pipe
    (``| tee run.log``) Python block-buffers it and the two processes' messages
    come out interleaved in the wrong order.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001  (not a TextIOWrapper: nothing to do)
            pass


def resolve_worker_device(event_workers: int, worker_device: str) -> str:
    if worker_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("--worker-device must be one of: auto, cpu, gpu")
    if worker_device == "auto":
        return "cpu" if event_workers > 1 else "gpu"
    return worker_device


def pin_this_process_to_cpu(context: str = "") -> None:
    """Actually move THIS process onto the JAX CPU backend.

    ``os.environ["JAX_PLATFORMS"] = "cpu"`` alone is a no-op here: jax snapshots
    its platform configuration when it is imported (which happened at the top of
    this module, via src.tape_jax), so the environment variable only reaches
    processes spawned afterwards.  ``jax.config.update("jax_platform_name")`` is
    honoured as long as no backend has been initialised yet, which is the case
    while the driver has not touched a jnp array.  The env vars are still set,
    because that is what spawned children inherit.
    """
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    try:
        import jax

        jax.config.update("jax_platform_name", "cpu")
        devices = [str(d) for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not pin this process to the JAX CPU backend: {exc}")
        return
    if any("cuda" in d.lower() or "gpu" in d.lower() for d in devices):
        print(
            f"WARNING: this process is still on {devices} after asking for the CPU "
            "backend (a jnp array was created before the switch)."
            + (f" [{context}]" if context else "")
        )


def configure_worker_device(event_workers: int, worker_device: str) -> str:
    resolved_device = resolve_worker_device(event_workers, worker_device)
    if resolved_device == "cpu":
        pin_this_process_to_cpu("event workers")
        print("Worker device: CPU; JAX forced to CPU for event workers to avoid multi-process GPU OOM.")
    elif event_workers > 1:
        print("Worker device: GPU; warning: multiple GPU worker processes may OOM on a single small GPU.")
    return resolved_device


def _gpu_memory_used_mib() -> float | None:
    """Total device memory currently in use, or None if nvidia-smi is missing."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def round_up(value: int, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(-(-int(value) // int(multiple)) * int(multiple))


def batch_shape_key(data: dict, inversion_options: list[str]) -> tuple[int, int]:
    """Cheap ``(N_pol, N_ar)`` estimate used only to ORDER and group events.

    ``prepare_batch`` does the real, exact bucketing (it has to build the
    station-intersected observation matrices to know the true counts); this
    proxy just counts the stations that feed each observation type so that
    sorted events land in shape-contiguous chunks and a chunk usually compiles
    a single XLA program.  Being wrong only costs an extra compile.
    """
    n_pol = 0
    n_ar = 0
    options = set(inversion_options or [])
    for key, entry in data.items():
        if not isinstance(entry, dict) or "Stations" not in entry:
            continue
        if options and key not in options:
            continue
        names = np.asarray(entry["Stations"].get("Name", []), dtype=object).reshape(-1)
        lowered = key.lower()
        if "polarity" in lowered and "prob" not in lowered:
            n_pol += int(names.size)
        elif "amplituderatio" in lowered:
            n_ar += int(names.size)
    return n_pol, n_ar


def _attach_compile_counter() -> list[str]:
    """Collect one entry per XLA compilation in this process (see M1 check 8a).

    Everything else jax compiles (``jit(add)``, ``jit(convert_element_type)``,
    ...) is one-off warm-up of tiny elementwise kernels; the number that matters
    is how often the three *batched programs* are compiled, which is one set per
    distinct padded shape + batch width.
    """
    import logging

    import jax

    records: list[str] = []

    # jax_log_compiles emits one WARNING per traced/compiled program, from two
    # different modules. We want the count, not the noise, so a filter records
    # the compilations and drops the chatter before it reaches any handler.
    # A logger-level filter only sees records logged through that exact logger,
    # so both emitters are filtered; anything else they log (real warnings) is
    # left alone.
    chatter_by_logger = {
        "jax._src.dispatch": (
            "Finished tracing + transforming",
            "Finished jaxpr to MLIR module conversion",
            "Finished XLA compilation",
        ),
        "jax._src.interpreters.pxla": ("Compiling ",),
    }

    class _CompileFilter(logging.Filter):
        def __init__(self, prefixes):
            super().__init__()
            self._prefixes = prefixes

        def filter(self, record):  # noqa: A003
            try:
                message = record.getMessage()
            except Exception:  # noqa: BLE001
                return True
            if "Finished XLA compilation" in message:
                records.append(message)
            return not message.startswith(self._prefixes)

    for logger_name, prefixes in chatter_by_logger.items():
        logging.getLogger(logger_name).addFilter(_CompileFilter(prefixes))
    jax.config.update("jax_log_compiles", True)
    return records


def _count_program_compiles(records: list[str]) -> int:
    return sum(
        1
        for message in records
        if any(f"jit({name})" in message for name in GPU_PROGRAM_NAMES)
    )


# ==============================================================================
# BATCHED GPU PATH (M2, plan section 4 / M1_STATUS.md section 6)
#
# Process layout:
#   * THE DRIVER process stays x64-enabled (fp64) and does all CPU work --
#     loading/filtering, Q metrics, CSV + pickles, and the unbatched fallback
#     inversion for events the batched sampler rejects or fails on.  It forces
#     itself onto the CPU backend once the worker is up, so only one process
#     ever touches the GPU.
#   * ONE persistent spawned subprocess disables jax_enable_x64 immediately
#     after import (the flag is process-global) and then serves every chunk for
#     the rest of the run.  It has to be persistent: the batched module's XLA
#     program cache lives in module globals, so a process per chunk would pay
#     ~20 s of compile per chunk and throw away the whole M1 win.
#
# IPC: one spawn-context Queue in each direction.  Payloads are pickled event
# DataDicts (~50 KB) on the way in and pickled InversionResults (~1 MB per event
# at P=2000 x 4 chains) on the way back -- small enough that a queue is simpler
# and faster than spooling temp files, and the driver needs the objects in
# memory anyway to post-process them.
# ==============================================================================


def _gpu_worker_main(task_q, result_q, cfg: dict) -> None:
    """Persistent fp32 GPU worker (spawned; see the section header above)."""
    import traceback as _traceback

    line_buffer_stdout()  # keep the child's lines in order with the driver's

    try:
        # The driver pops JAX_PLATFORMS/JAX_PLATFORM_NAME from os.environ before
        # spawning this child (and only then forces ITSELF onto the CPU
        # backend), so jax picks the GPU here. Popping them in this function
        # would be too late: spawn re-imports this module, which imports jax,
        # and jax snapshots its platform config at import.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if cfg.get("mem_fraction"):
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg["mem_fraction"])

        import jax

        # jax_enable_x64 is process-global and src.tape_jax switched it ON when
        # this module imported InversionBlackJAX. Switch it off now, before the
        # first array exists -- that is what makes the fp32 sampler fp32.
        jax.config.update("jax_enable_x64", False)
        from .inversion_blackjax_batched import run_batched

        compiles = _attach_compile_counter() if cfg.get("log_compiles") else None
        devices = [str(d) for d in jax.devices()]
        result_q.put(
            {
                "type": "ready",
                "devices": devices,
                "x64": bool(jax.config.jax_enable_x64),
                "pid": os.getpid(),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        result_q.put(
            {
                "type": "ready_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_q.get()
        if task is None:
            break
        chunk_id = task["chunk_id"]
        events = task["events"]
        n_compiles_before = len(compiles) if compiles is not None else 0
        buckets: list[str] = []

        # Per-stage lines are the bulk of the console output (~30 per bucket).
        # Keep the first, every Nth and the final stage; everything that is not
        # a stage line (init+compile, completed, warnings) always prints.
        stage_every = max(1, int(cfg.get("stage_log_every", 1) or 1))

        def emit(message: str, _chunk_id=chunk_id, _buckets=buckets) -> None:
            if message.startswith("[bucket") and "init+compile" in message:
                _buckets.append(message.split("]")[0].lstrip("["))
            if stage_every > 1:
                match = _STAGE_LINE_RE.search(message)
                if match is not None:
                    stage_no = int(match.group("stage"))
                    done, total = match.group("done"), match.group("total")
                    if not (
                        stage_no == 1
                        or stage_no % stage_every == 0
                        or done == total  # last stage of the bucket
                    ):
                        return
            print(f"[gpu chunk {_chunk_id}] {message}", flush=True)

        t0 = time.time()
        payload = {"type": "result", "chunk_id": chunk_id, "results": {}, "failures": {}}
        try:
            inv = Inversion(events[0][1], **cfg["inv_kwargs"])
            run_kwargs = {}
            if cfg.get("per_event_seeds"):
                # Independent RNG streams per event inside one batch. Without
                # this every event of a chunk shares chain seeds, initial
                # particles and SMC keys (common random numbers), which
                # correlates the Monte-Carlo error of neighbouring events.
                import inspect

                if "per_event_seeds" in inspect.signature(run_batched).parameters:
                    run_kwargs["per_event_seeds"] = True
                else:
                    print(
                        "[gpu worker] WARNING: this SMTI checkout's run_batched has no "
                        "per_event_seeds argument; every event in a batch will share "
                        "chain seeds and initial particles (correlated MC error).",
                        flush=True,
                    )
            results, failures = run_batched(
                inv,
                events,
                smc_dtype=cfg["smc_dtype"],
                max_batch=cfg["max_batch"],
                pad_multiple=cfg.get("pad_multiple", 16),
                return_failures=True,
                progress_callback=emit,
                **run_kwargs,
            )
            payload["results"] = results
            payload["failures"] = failures
        except NotImplementedError as exc:
            # Scope guard (persistent SMC / NUTS / station_smooth / dc=True ...).
            # Every event of this chunk goes back to the unbatched fp64 driver.
            payload["unsupported"] = f"{type(exc).__name__}: {exc}"
            print(f"[gpu chunk {chunk_id}] unsupported configuration: {exc}", flush=True)
        except BaseException as exc:  # noqa: BLE001
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = _traceback.format_exc()
            print(f"[gpu chunk {chunk_id}] FAILED: {exc}", flush=True)
        payload["wall"] = time.time() - t0
        payload["buckets"] = buckets
        if compiles is None:
            payload["compiles"] = None
            payload["program_compiles"] = None
        else:
            new_records = compiles[n_compiles_before:]
            payload["compiles"] = len(new_records)
            payload["program_compiles"] = _count_program_compiles(new_records)
        payload["gpu_mib"] = _gpu_memory_used_mib()
        result_q.put(payload)

    if compiles is not None:
        print(
            f"[gpu worker] XLA compilations: {len(compiles)} total, "
            f"{_count_program_compiles(compiles)} of them batched programs "
            f"({'/'.join(GPU_PROGRAM_NAMES)})",
            flush=True,
        )
    result_q.put(
        {
            "type": "closed",
            "compiles": None if compiles is None else len(compiles),
            "program_compiles": None if compiles is None else _count_program_compiles(compiles),
        }
    )


class GpuBatchWorker:
    """Handle for the single persistent fp32 GPU subprocess."""

    def __init__(self, cfg: dict):
        self.ctx = mp.get_context("spawn")
        self.cfg = cfg
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_gpu_worker_main,
            args=(self.task_q, self.result_q, cfg),
            name="smti-gpu-worker",
        )
        # Daemonic: a non-daemon child blocked on task_q.get() makes the
        # interpreter hang forever at exit (multiprocessing joins it) while it
        # keeps holding the CUDA context on a shared GPU. With daemon=True an
        # unhandled driver exit terminates it instead.
        self.proc.daemon = True
        self.devices: list[str] = []
        self.total_compiles: int | None = None
        self.total_program_compiles: int | None = None

    def start(self, timeout: float = 900.0) -> None:
        self.proc.start()
        message = self._get(timeout=timeout)
        if message.get("type") != "ready":
            raise RuntimeError(
                "fp32 GPU worker failed to start: "
                f"{message.get('error')}\n{message.get('traceback', '')}"
            )
        self.devices = message["devices"]
        print(
            f"fp32 GPU worker ready (pid {message['pid']}, devices={self.devices}, "
            f"jax_enable_x64={message['x64']})",
            flush=True,
        )
        if not any("cuda" in d.lower() or "gpu" in d.lower() for d in self.devices):
            raise RuntimeError(
                f"fp32 GPU worker came up on {self.devices}; expected a CUDA device. "
                "Unset JAX_PLATFORMS / JAX_PLATFORM_NAME / CUDA_VISIBLE_DEVICES "
                "before using --worker-device gpu (the README's step-4 recipe "
                "starts with JAX_PLATFORMS=cpu, which the worker would inherit)."
            )

    def _get(self, timeout: float | None = None) -> dict:
        """Blocking receive that notices a dead OR WEDGED worker.

        ``timeout`` is a real wall-clock budget: a worker that is alive but
        stuck (CUDA hang, XLA deadlock, a tempering loop that never ends) used
        to block the driver forever, because the only exit from this loop was
        the worker dying.
        """
        waited = 0.0
        poll = 1.0
        while True:
            try:
                return self.result_q.get(timeout=poll)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(
                        f"fp32 GPU worker died (exit code {self.proc.exitcode})"
                    ) from None
                waited += poll
                if timeout is not None and waited >= timeout:
                    raise RuntimeError(
                        f"fp32 GPU worker is alive but produced nothing for {waited:.0f} s "
                        f"(budget {timeout:.0f} s); treating it as wedged"
                    ) from None

    def submit(self, chunk_id: int, events: list[tuple[str, dict]]) -> None:
        self.task_q.put({"chunk_id": chunk_id, "events": events})

    def collect(self, timeout: float | None = None) -> dict:
        return self._get(timeout=timeout)

    def kill(self) -> None:
        """Hard stop for a wedged worker; safe to call twice."""
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=30)

    def close(self) -> None:
        if self.proc.pid is None:  # never started
            return
        try:
            if self.proc.is_alive():
                self.task_q.put(None)
                message = self._get(timeout=30.0)
                self.total_compiles = message.get("compiles")
                self.total_program_compiles = message.get("program_compiles")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: fp32 GPU worker shutdown: {exc}")
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            print("Warning: fp32 GPU worker did not exit; terminating.")
            self.kill()


def event_log_path(event_log_dir: str | None, event_id: str) -> str | None:
    if not event_log_dir:
        return None
    os.makedirs(event_log_dir, exist_ok=True)
    return os.path.join(event_log_dir, f"{event_id}.log")


def gpu_submissions(
    preps: list[dict], batch_events: int, submit_cap: int
) -> list[list[dict]]:
    """Split shape-sorted events into ONE submission per padded shape.

    A submission is handed to ``run_batched`` whole, so that ``max_batch``
    chunking with repeat padding happens *inside* the batched sampler: a shape
    group of 5 events at ``--gpu-batch 4`` becomes chunks of B=16 and B=16
    (the second repeat-padded) sharing one compiled program, instead of the
    B=16 + B=4 the driver used to produce, which compiled a second program set
    (~16 s) for every shape whose event count is not a multiple of --gpu-batch.

    ``submit_cap`` bounds one submission so a large same-shape group does not
    serialise all of its post-processing behind one worker round trip.  A
    trailing piece of ``<= batch_events`` events is merged into its predecessor,
    because such a piece would again run at its own narrow width.
    """
    submit_cap = max(int(submit_cap), int(batch_events))
    groups: list[list[dict]] = []
    for prep in preps:
        if groups and prep["bucket_key"] == groups[-1][0]["bucket_key"]:
            groups[-1].append(prep)
        else:
            groups.append([prep])

    submissions: list[list[dict]] = []
    for group in groups:
        pieces = [group[i:i + submit_cap] for i in range(0, len(group), submit_cap)]
        if len(pieces) > 1 and len(pieces[-1]) <= batch_events:
            pieces[-2].extend(pieces.pop())
        submissions.extend(pieces)
    return submissions


def terminate_pool_workers(executor, label: str, grace_seconds: float = 5.0) -> None:
    """Shut an executor down and make sure ITS OWN children are really gone.

    Deliberately not ``process_cleanup.cleanup_process_pool``: that one also
    targets every recursive child of this process, which here includes the fp32
    GPU worker (and the loader children) -- killing those mid-run would be far
    worse than a leaked post-processing worker.  Only the pids the executor
    itself owns are signalled, so a wedged or crashed post worker cannot survive
    the pool it belongs to.
    """
    processes = getattr(executor, "_processes", None) or {}
    pids = [p.pid for p in processes.values() if getattr(p, "pid", None)]
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: {label} shutdown: {exc}", flush=True)

    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    for signum in (signal.SIGTERM, signal.SIGKILL):
        remaining = [pid for pid in pids if _alive(pid)]
        if not remaining:
            return
        for pid in remaining:
            try:
                os.kill(pid, signum)
            except (ProcessLookupError, PermissionError):
                pass
        if signum is signal.SIGKILL:
            return
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.2)


def _post_pool_init() -> None:
    """Initializer for the post-processing pool's children.

    The GPU belongs to the one batched sampler worker, and a post-processing job
    may run a full unbatched fp64 inversion, so every child has to be on the JAX
    CPU backend before it touches jax.  Under spawn the child re-imports this
    module first, which imports jax via src.inversion_blackjax -- but it
    inherits ``JAX_PLATFORMS=cpu`` from the driver
    (``pin_this_process_to_cpu`` sets it), so that import already lands on the
    CPU.  This call makes it true even if the env var was lost.
    """
    pin_this_process_to_cpu("post-processing pool worker")
    line_buffer_stdout()


class PostProcessPool:
    """Bounded spawn pool that runs event post-processing off the GPU path.

    ``submit`` blocks only while ``pending_multiplier * max_workers`` jobs are
    already outstanding, so the driver spends its time collecting and submitting
    GPU chunks instead of writing CSVs, and never holds more than that many
    pickled events alive.  Finished jobs are handed to
    ``on_done(info, report, error)``.

    ``task_fn`` is the picklable, module-level function each job runs; it is
    supplied by the caller because what "post-processing" means is
    project-specific.

    Every submitted job reaches ``on_done`` exactly once -- including one whose
    worker raised, one whose worker died and took the rest of its pool
    generation with it, one that hung, and one that was in a child when a
    terminal Ctrl-C hit the whole process group -- so the caller's results list
    still ends up with one entry per event.

    Nothing here ever blocks without a deadline (``job_timeout``), and nothing
    that has already killed a worker process is re-run in the driver: after the
    restart budget is spent the remaining jobs get a short-lived subprocess of
    their own, so a segfaulting or OOM-killed job costs one event instead of the
    whole run's results and the GPU worker.

    Parameters
    ----------
    job_timeout : float
        Wall-clock budget for ONE post-processing job before the child holding
        it is declared wedged, its event failed and the pool torn down.  Same
        reasoning as the GPU chunk timeouts on the sampler side: without it a
        single stuck job (an fp64 fallback that will not converge, a KDE on a
        pathological posterior, a native lock) blocks submit() and, because
        close() sits in the driver's finally, blocks the Ctrl-C exit path too --
        with the GPU still held.
    poll_seconds : float
        How often a blocking harvest wakes up to re-check those deadlines.
    max_tasks_per_child : int | None
        Recycle a post-processing worker after this many jobs (the loader pool
        uses the same trick): arviz/KDE/jax-CPU memory accumulates over hundreds
        of events, and a worker that grows into the OOM killer looks like a
        crash.
    """

    def __init__(
        self,
        max_workers: int,
        on_done,
        task_fn,
        pending_multiplier: int = 2,
        max_restarts: int = 2,
        label: str = "post-processing pool",
        job_timeout: float = 3600.0,
        poll_seconds: float = 30.0,
        max_tasks_per_child: int | None = 50,
        max_isolated_failures: int = 3,
    ):
        if max_workers < 1:
            raise ValueError("post pool max_workers must be >= 1")
        self.max_workers = int(max_workers)
        self.max_pending = max(1, self.max_workers * max(1, int(pending_multiplier)))
        self.on_done = on_done
        self.task_fn = task_fn
        self.label = label
        self.max_restarts = int(max_restarts)
        self.job_timeout = float(job_timeout)
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.max_tasks_per_child = max_tasks_per_child
        self.max_isolated_failures = int(max_isolated_failures)
        self.ctx = mp.get_context("spawn")
        self.pending: dict = {}   # future -> job dict (info / task / attempts)
        self.submitted = 0        # distinct jobs handed out (retries not counted)
        self.restarts = 0         # broken executors replaced
        self.inline = 0           # jobs the driver had to run itself
        self.isolated = 0         # jobs given a subprocess of their own
        self.expired = 0          # jobs killed for exceeding job_timeout
        self.max_outstanding = 0  # high-water mark, reported in the run summary
        self.executor = None
        self.pool_lost = False    # a worker died and the restart budget is gone
        self.stopped = False      # abort(): no more work of any kind
        self._isolated_failures = 0
        self._progress_at = time.monotonic()  # last time any job resolved
        self._started = self._open()
        # A driver that dies without running its finally (an uncaught exception,
        # SIGTERM, sys.exit) must not leave children mid-write in the output
        # tree.  threading._register_atexit runs BEFORE concurrent.futures'
        # own _python_exit, which would otherwise join the workers -- i.e. wait
        # for the very jobs we want gone; plain atexit is the fallback.
        self._exit_hook_registered = False
        try:
            threading._register_atexit(self._atexit_cleanup)
            self._exit_hook_registered = True
        except Exception:  # noqa: BLE001 - private API, best effort
            pass
        atexit.register(self._atexit_cleanup)

    def _open(self) -> bool:
        """Start a fresh executor. Workers spawn lazily, on the first submit."""
        kwargs = {
            "max_workers": self.max_workers,
            "mp_context": self.ctx,
            "initializer": _post_pool_init,
        }
        if self.max_tasks_per_child and sys.version_info >= (3, 11):
            kwargs["max_tasks_per_child"] = int(self.max_tasks_per_child)
        try:
            self.executor = ProcessPoolExecutor(**kwargs)
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: {self.label}: could not start ({exc}); post-processing "
                "runs in the driver."
            )
            self.executor = None
            return False

    def submit(self, task: tuple, info: dict) -> None:
        """Queue one post-processing job; blocks only when the bound is full."""
        while len(self.pending) >= self.max_pending:
            self._harvest(block=True)
        self.submitted += 1
        self._dispatch({"info": info, "task": task, "attempts": 0})
        self._harvest(block=False)  # free: integrate whatever already finished

    def drain(self) -> None:
        """Integrate every outstanding job; nothing is left after this returns.

        Bounded: a pool that produces nothing for ``job_timeout`` is torn down
        by ``_expire``, so a wedged child cannot park the driver here forever
        (which, since ``close()`` runs in the driver's ``finally``, used to
        make the run unkillable while the GPU was still held).
        """
        while self.pending:
            self._harvest(block=True)

    def close(self) -> None:
        try:
            self.drain()
        finally:
            executor, self.executor = self.executor, None
            if executor is not None:
                try:
                    executor.shutdown(wait=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: {self.label} shutdown: {exc}")
            self._unregister_atexit()

    def abort(self, reason: str) -> None:
        """Give up now: kill the workers, fail whatever is still outstanding.

        Used on SIGTERM, where the point is to be gone before the scheduler's
        SIGKILL arrives -- draining could take job_timeout.
        """
        self.stopped = True
        executor, self.executor = self.executor, None
        if executor is not None:
            terminate_pool_workers(executor, self.label)
        stragglers = list(self.pending.items())
        self.pending.clear()
        for future, job in stragglers:
            future.cancel()
            self.on_done(job["info"], None, RuntimeError(f"{self.label}: {reason}"))
        if stragglers:
            print(
                f"WARNING: {self.label}: {reason}; {len(stragglers)} post-processing "
                "job(s) were abandoned.",
                flush=True,
            )
        self._unregister_atexit()

    def _atexit_cleanup(self) -> None:
        """Interpreter is going down and close() never ran: reap the children."""
        executor, self.executor = self.executor, None
        if executor is None:
            return
        print(
            f"WARNING: {self.label}: driver exiting with workers still alive; "
            "killing them so they cannot keep writing output directories.",
            flush=True,
        )
        terminate_pool_workers(executor, self.label)

    def _unregister_atexit(self) -> None:
        try:
            atexit.unregister(self._atexit_cleanup)
        except Exception:  # noqa: BLE001
            pass

    def _dispatch(self, job: dict) -> None:
        if self.executor is None:
            if self._started and not self.stopped:
                # The pool worked once and then died for good: never hand the
                # job that may have killed it to the driver process.
                self._resolve_degraded(job, RuntimeError(f"{self.label}: no worker pool"))
            elif self.stopped:
                self.on_done(job["info"], None, RuntimeError(f"{self.label}: aborted"))
            else:
                # The pool could never be started at all (no worker has ever
                # run, so nothing is under suspicion): same as --post-workers 0.
                self._run_inline(job)
            return
        job["attempts"] += 1
        try:
            future = self.executor.submit(self.task_fn, job["task"])
        except Exception as exc:  # noqa: BLE001 - broken/shut-down executor
            print(
                f"WARNING: {self.label}: submit failed for "
                f"{job['info']['event_id']} ({exc})."
            )
            if job["attempts"] <= self.max_restarts and self._replace_pool():
                self._dispatch(job)
            else:
                self._resolve_degraded(job, exc)
            return
        job["submitted_at"] = time.monotonic()
        self.pending[future] = job
        self.max_outstanding = max(self.max_outstanding, len(self.pending))

    def _note_progress(self) -> None:
        """One job resolved: the pool is alive, restart the stall clock."""
        self._progress_at = time.monotonic()

    def _wait_timeout(self) -> float:
        """Wake up in time to check the stall clock (or just to poll)."""
        left = self._progress_at + self.job_timeout - time.monotonic()
        return max(0.05, min(self.poll_seconds, left))

    def _harvest(self, block: bool) -> None:
        """Integrate finished jobs; with ``block`` wait for at least one."""
        if not self.pending:
            return
        if block:
            done, _ = wait(
                list(self.pending),
                return_when=FIRST_COMPLETED,
                timeout=self._wait_timeout(),
            )
        else:
            done = [future for future in list(self.pending) if future.done()]
        orphaned: list = []
        for future in done:
            job = self.pending.get(future)
            if job is None:  # already resolved by _recover/_expire
                continue
            try:
                report, error = future.result(), None
            except BrokenProcessPool as exc:
                # Not this job's fault: a worker died and the whole generation
                # went with it. Rescued below instead of failed here.
                self.pending.pop(future, None)
                orphaned.append((job, exc))
                continue
            except BaseException as exc:  # noqa: BLE001
                # BaseException, not Exception: a spawned worker hit by the
                # terminal's Ctrl-C (SIGINT goes to the whole foreground process
                # group) sends KeyboardInterrupt back as this future's result.
                # That is this job failing, not the driver being interrupted --
                # letting it escape here aborted the run and dropped every
                # outstanding job, including ones that had already succeeded.
                # A real interrupt of the driver is raised by wait() above,
                # outside this try, and still propagates.
                report, error = None, exc
            # Pop only once the result is in hand, so an exception on the way
            # out can never lose a job that is no longer in `pending`.
            self.pending.pop(future, None)
            self._note_progress()
            self.on_done(job["info"], report, error)
        if orphaned:
            self._recover(orphaned, "a worker process died")
        elif block:
            self._expire()

    def _expire(self) -> None:
        """Break a stalled pool: nothing has finished for a whole job_timeout.

        The clock is "no job resolved", not "this job was submitted long ago":
        a queue of slow-but-healthy fp64 fallbacks keeps resetting it, one
        wedged worker among several does not stop the others, and only a pool
        that has stopped producing anything at all is torn down.  The
        longest-outstanding job is the suspect and is failed; every other
        outstanding job is re-run on the fresh pool by ``_recover``.
        """
        if not self.pending:
            return
        if time.monotonic() - self._progress_at < self.job_timeout:
            return
        future = min(
            self.pending, key=lambda f: self.pending[f].get("submitted_at", 0.0)
        )
        job = self.pending.pop(future)
        job["fatal"] = True  # never re-run a job that already wedged a child
        self.expired += 1
        self._note_progress()
        self._recover(
            [
                (
                    job,
                    TimeoutError(
                        f"post-processing of {job['info']['event_id']} produced "
                        f"nothing for {self.job_timeout:.0f} s"
                    ),
                )
            ],
            f"no post-processing job finished for {self.job_timeout:.0f} s",
        )

    def _recover(self, orphaned: list, reason: str) -> None:
        """A worker died or hung: rescue the jobs of that pool generation."""
        # Every other outstanding future of this generation is resolved too:
        # keep the ones that did finish, treat the rest as orphans.
        for future in list(self.pending):
            job = self.pending[future]
            try:
                report = future.result(timeout=0)
            except BaseException as exc:  # noqa: BLE001
                self.pending.pop(future, None)
                orphaned.append((job, exc))
                continue
            self.pending.pop(future, None)
            self._note_progress()
            self.on_done(job["info"], report, None)
        print(
            f"WARNING: {self.label}: {reason}; "
            f"{len(orphaned)} job(s) lost their result.",
            flush=True,
        )
        replaced = self._replace_pool()
        for job, exc in orphaned:
            if replaced and not job.get("fatal") and job["attempts"] <= self.max_restarts:
                self._dispatch(job)
            else:
                self._resolve_degraded(job, exc)

    def _replace_pool(self) -> bool:
        """Throw away a broken executor and start a fresh one (bounded count)."""
        executor, self.executor = self.executor, None
        if executor is not None:
            # shutdown(wait=False) alone leaves a *wedged* child running: this
            # also SIGTERM/SIGKILLs the workers this executor owns.
            terminate_pool_workers(executor, self.label)
        if self.restarts >= self.max_restarts:
            self.pool_lost = True
            print(
                f"WARNING: {self.label}: already restarted {self.restarts} time(s); "
                "the remaining post-processing runs in a one-shot subprocess per "
                "event (slower, but a crashing job cannot take the driver down)."
            )
            return False
        self.restarts += 1
        if not self._open():
            self.pool_lost = True
            return False
        print(f"{self.label}: restarted ({self.restarts}/{self.max_restarts}).")
        return True

    def _resolve_degraded(self, job: dict, exc) -> None:
        """No usable pool: salvage the job in its own process, or fail it."""
        if self.stopped or job.get("fatal"):
            self.on_done(job["info"], None, exc)
            return
        if self._isolated_failures >= self.max_isolated_failures:
            self.on_done(
                job["info"],
                None,
                RuntimeError(
                    f"{self.label}: post-processing disabled after "
                    f"{self._isolated_failures} consecutive isolated-subprocess "
                    f"failures (last pool error: {exc})"
                ),
            )
            return
        self._run_isolated(job)

    def _run_isolated(self, job: dict) -> None:
        """Run ONE job in a subprocess of its own, then throw that process away.

        The degraded path after the restart budget is spent.  The event is still
        finalized (it already cost GPU time), but a native crash or an OOM kill
        only fails that event: the driver keeps `results`, the GPU worker handle
        and its finally blocks.
        """
        self.isolated += 1
        executor = None
        future = None
        try:
            executor = ProcessPoolExecutor(
                max_workers=1, mp_context=self.ctx, initializer=_post_pool_init
            )
            future = executor.submit(self.task_fn, job["task"])
            report, error = future.result(timeout=self.job_timeout), None
        except BaseException as exc:  # noqa: BLE001
            report, error = None, exc
            local = future is None or not future.done()
            if local and isinstance(exc, (KeyboardInterrupt, SystemExit)):
                # The interrupt hit the driver, not the job: report the event
                # (exactly-once) and let the abort continue.
                if executor is not None:
                    terminate_pool_workers(executor, f"{self.label} (isolated)")
                self.on_done(job["info"], None, exc)
                raise
        finally:
            if executor is not None:
                terminate_pool_workers(executor, f"{self.label} (isolated)")
        self._isolated_failures = 0 if error is None else self._isolated_failures + 1
        self._note_progress()
        self.on_done(job["info"], report, error)

    def _run_inline(self, job: dict) -> None:
        """Only when no worker has ever run: do the job here (as --post-workers 0)."""
        self.inline += 1
        try:
            report, error = self.task_fn(job["task"]), None
        except BaseException as exc:  # noqa: BLE001
            report, error = None, exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                self.on_done(job["info"], None, exc)
                raise
        self._note_progress()
        self.on_done(job["info"], report, error)


def gpu_scope_problems(
    amp_ratio_noise_mode: str | None, dc: bool, mcmc_kernel: str | None
) -> list[str]:
    """Configuration reasons the batched sampler would reject EVERY event.

    These are checked in the driver, before the GPU worker is started, because
    the alternative is a run that holds the GPU for hours while inverting every
    event serially in the driver on the CPU (measured: 59 ev/h, i.e. slower
    than not passing --worker-device gpu at all).
    """
    problems = []
    if amp_ratio_noise_mode != "global":
        problems.append(
            f"--amp-ratio-noise-mode {amp_ratio_noise_mode!r} "
            "(the batched sampler supports 'global' only)"
        )
    if dc:
        problems.append("DC_CONSTRAINT = True (the batched sampler supports dc=False only)")
    if mcmc_kernel not in {"rmh", "irmh", "mwg_irmh"}:
        problems.append(
            f"MCMC_KERNEL = {mcmc_kernel!r} "
            "(batched sampler supports 'rmh', 'irmh', 'mwg_irmh' only)"
        )
    return problems


def run_event_batch_gpu(
    events_dat: list[str],
    *,
    inv_kwargs: dict,
    output_dir: str,
    make_load_task,
    load_task_fn,
    make_post_task,
    post_task_fn,
    event_log_dir: str | None = None,
    gpu_batch_events: int = 6,
    load_workers: int = 8,
    mem_fraction: str | None = None,
    log_compiles: bool = True,
    allow_cpu_fallback: bool = False,
    smc_dtype: str = "float32",
    submit_cap: int = 32,
    stage_log_every: int = 5,
    max_events_per_worker: int = 50,
    random_seed: int = 0,
    pad_multiple: int = 16,
    post_workers: int = 4,
    chunk_timeout_per_event: float = 900.0,
    chunk_timeout_min: float = 1800.0,
    post_job_timeout: float = 3600.0,
    post_poll_seconds: float = 30.0,
    post_tasks_per_child: int | None = 50,
) -> list[dict]:
    """Batched fp32 GPU mode: one persistent worker, shape-grouped, overlapped.

    Parameters
    ----------
    inv_kwargs : dict
        The complete ``InversionBlackJAX`` keyword arguments (everything except
        the event data), priors included.  Must carry ``num_chains``, ``dc``,
        ``mcmc_kernel`` and ``amp_ratio_noise_mode``: the batch width is
        ``gpu_batch_events * num_chains`` and the other three decide whether the
        batched sampler can run this configuration at all.
    make_load_task, load_task_fn, make_post_task, post_task_fn
        The project hooks; see the module docstring.
    gpu_batch_events : int
        Events per batch; B = ``gpu_batch_events * num_chains``.
    smc_dtype : str
        Sampling precision inside the worker; fp32 is the decisive lever on an
        A40 (9.3x/stage, M0).
    load_workers : int
        CPU processes pre-loading event inputs for the GPU queue.
    pad_multiple : int
        Station-count padding for bucket shapes.  Passed to
        ``run_batched(pad_multiple=...)`` AND used for the driver-side shape
        grouping, so the two always agree.  M3 shape survey over 2116 events:
        16 gives 42 shapes (~11 min compile, 9.7% masked rows) vs 129
        shapes/~34 min/4.4% at 8 -- see M3_STATUS.md section 3.
    submit_cap : int
        Upper bound on the events handed to the worker in ONE submission.
        Submissions are whole shape groups so that run_batched can do its own
        max_batch chunking with repeat padding (one XLA program per shape
        instead of one per remainder); the cap only stops a huge same-shape
        group from serialising the post-processing pipeline and from holding
        every result in the worker at once.
    stage_log_every : int
        Console verbosity of the sampler: print stage 1, every Nth stage and the
        last one of each bucket (1 = every stage).  A bucket takes ~30 stages,
        so on a 450-event run the default turns ~3400 stage lines into ~700.
    chunk_timeout_per_event, chunk_timeout_min : float
        Wall-clock budget for ONE submission before the worker is declared
        wedged.  Scaled per event, with ``chunk_timeout_min`` as the floor: a
        hung CUDA/XLA call used to block the driver forever (a *dead* worker was
        always detected).
    post_workers : int
        Post-processing (ArviZ summary, Q metrics, CSV + pickles, and in the
        worst case a FULL fp64 CPU fallback inversion) used to run serially in
        the driver between two GPU submissions, i.e. on the GPU critical path:
        M2 measured a post_wall comparable to gpu_wall, and a single fallback is
        minutes.  These spawned processes take it off that path.  0 = the old
        serial behaviour.
    post_job_timeout, post_poll_seconds, post_tasks_per_child
        Passed straight to :class:`PostProcessPool`.
    """
    line_buffer_stdout()  # driver and worker share stdout; keep them in order
    if gpu_batch_events < 1:
        raise ValueError("--gpu-batch must be >= 1")
    inv_kwargs = dict(inv_kwargs)

    # ---- configuration scope: refuse before we take the GPU -----------------
    problems = gpu_scope_problems(
        inv_kwargs.get("amp_ratio_noise_mode"),
        bool(inv_kwargs.get("dc")),
        inv_kwargs.get("mcmc_kernel"),
    )
    if problems and not allow_cpu_fallback:
        raise SystemExit(
            "--worker-device gpu cannot use the batched sampler with this "
            "configuration:\n  - " + "\n  - ".join(problems) + "\n"
            "Every event would fall back to a serial unbatched CPU inversion inside "
            "the driver (~59 events/hour) while still holding the GPU.\n"
            "Fix: add --amp-ratio-noise-mode global to use the GPU, or use "
            "--worker-device cpu for the CPU pool (~105 ev/h).\n"
            "Pass --allow-cpu-fallback if you really want the slow serial path."
        )
    if problems:
        print(
            "WARNING: --allow-cpu-fallback given and the configuration is outside the "
            "batched sampler's scope (" + "; ".join(problems) + "); every event will "
            "be inverted serially on the CPU inside the driver while the GPU worker "
            "sits idle."
        )

    total = len(events_dat)
    num_chains = int(inv_kwargs["num_chains"])
    max_batch = int(gpu_batch_events) * num_chains
    if event_log_dir is None:
        event_log_dir = os.path.join(output_dir, "event_logs")
    os.makedirs(event_log_dir, exist_ok=True)
    print(
        f"Worker device: GPU (batched). {total} events, {gpu_batch_events} events/batch "
        f"x {num_chains} chains = B<={max_batch}, smc_dtype={smc_dtype}, "
        f"random_seed={random_seed}."
    )
    print(f"Per-event logs: {event_log_dir}")

    # The spawned worker inherits os.environ. The README's step-4 recipe starts
    # with JAX_PLATFORMS=cpu, which would silently put the "GPU" worker on the
    # CPU backend, so strip those two here; the driver pins ITSELF to the CPU
    # backend right after the spawn (jax.config, not the env var -- see
    # pin_this_process_to_cpu).
    stripped = {
        key: os.environ.pop(key)
        for key in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")
        if key in os.environ
    }
    if stripped:
        print(
            "note: removing "
            + ", ".join(f"{k}={v!r}" for k, v in stripped.items())
            + " from the fp32 GPU worker's environment (it needs the CUDA backend)."
        )

    worker = GpuBatchWorker(
        {
            "smc_dtype": smc_dtype,
            "max_batch": max_batch,
            "inv_kwargs": inv_kwargs,
            "log_compiles": bool(log_compiles),
            "stage_log_every": int(stage_log_every),
            "mem_fraction": mem_fraction,
            "per_event_seeds": True,
            "pad_multiple": int(pad_multiple),
        }
    )
    try:
        worker.start()
    except BaseException:
        worker.close()  # never leave a half-started child holding the GPU
        raise

    results: list[dict] = []
    try:
        # EVERYTHING below runs under this try/finally: an exception in the load
        # / sort / submit phase used to skip worker.close(), leaking a child
        # that holds the CUDA context and blocks interpreter exit.
        results = _run_event_batch_gpu_inner(
            worker=worker,
            events_dat=events_dat,
            make_load_task=make_load_task,
            load_task_fn=load_task_fn,
            make_post_task=make_post_task,
            post_task_fn=post_task_fn,
            gpu_batch_events=gpu_batch_events,
            load_workers=load_workers,
            event_log_dir=event_log_dir,
            submit_cap=submit_cap,
            max_events_per_worker=max_events_per_worker,
            results=results,
            post_workers=post_workers,
            chunk_timeout_per_event=chunk_timeout_per_event,
            chunk_timeout_min=chunk_timeout_min,
            post_job_timeout=post_job_timeout,
            post_poll_seconds=post_poll_seconds,
            post_tasks_per_child=post_tasks_per_child,
        )
    finally:
        worker.close()
    return results


def _run_event_batch_gpu_inner(
    *,
    worker: "GpuBatchWorker",
    events_dat: list[str],
    make_load_task,
    load_task_fn,
    make_post_task,
    post_task_fn,
    gpu_batch_events: int,
    load_workers: int,
    event_log_dir: str,
    submit_cap: int,
    max_events_per_worker: int,
    results: list[dict],
    post_workers: int,
    chunk_timeout_per_event: float,
    chunk_timeout_min: float,
    post_job_timeout: float,
    post_poll_seconds: float,
    post_tasks_per_child: int | None,
) -> list[dict]:
    """Load -> shape-group -> sample -> post-process, with the worker already up."""
    total = len(events_dat)

    # The worker owns the GPU from here on; the driver (and every child it
    # spawns from now on, i.e. the loader pool) must stay on the CPU backend.
    pin_this_process_to_cpu("batched GPU driver")

    preps: list[dict] = []

    def iter_load_tasks(paths, offset=0):
        for index, dat_path in enumerate(paths, start=offset):
            yield make_load_task(index, total, dat_path)

    def record_loaded(prep: dict) -> None:
        if prep["kind"] == "ready":
            preps.append(prep)
        elif prep["kind"] == "skipped":
            results.append(prep["result"])
        else:
            print(f"Warning: could not load {prep['dat_path']}: {prep['error']}")
            results.append(
                {
                    "event_id": prep["event_id"],
                    "dat_path": prep["dat_path"],
                    "ok": False,
                    "error": prep["error"],
                    "traceback": prep.get("traceback"),
                    "log_path": prep.get("log_path"),
                }
            )

    t_load = time.time()
    if load_workers > 1 and total > 1:
        print(f"Loading {total} events with {load_workers} loader processes...")
        loaded_iter = bounded_process_pool_map(
            load_task_fn,
            iter_load_tasks(events_dat),
            max_workers=load_workers,
            mp_context=mp.get_context("spawn"),
            max_tasks_per_child=max(1, int(max_events_per_worker)),
            label="Step 4 GPU loader pool",
        )
    else:
        loaded_iter = (load_task_fn(task) for task in iter_load_tasks(events_dat))

    n_yielded = 0
    try:
        for prep in loaded_iter:
            n_yielded += 1
            record_loaded(prep)
    except BrokenProcessPool as exc:
        # A loader process died (OOM kill, native segfault, node hiccup). Do not
        # throw away the events already loaded and the worker's compile cache:
        # finish the remaining events serially in this process.
        remaining = events_dat[n_yielded:]
        print(
            f"WARNING: the loader pool died after {n_yielded}/{total} events ({exc}); "
            f"loading the remaining {len(remaining)} event(s) serially in the driver."
        )
        for task in iter_load_tasks(remaining, offset=n_yielded):
            record_loaded(load_task_fn(task))
    load_wall = time.time() - t_load
    print(
        f"Loaded {len(preps)} invertible events in {load_wall:.1f} s "
        f"({len(results)} skipped/failed at load)."
    )

    # Shape order -> one submission per padded shape (run_batched then does its
    # own max_batch chunking WITH repeat padding, so a shape compiles once).
    preps.sort(key=lambda p: (p["bucket_key"], p["shape_key"], p["event_id"]))
    chunks = gpu_submissions(preps, gpu_batch_events, submit_cap)
    shape_hist: dict[tuple[int, int], int] = {}
    for prep in preps:
        shape_hist[prep["bucket_key"]] = shape_hist.get(prep["bucket_key"], 0) + 1
    print(
        f"{len(chunks)} submission(s) over {len(shape_hist)} padded shape(s): "
        + ", ".join(f"(N_pol<={k[0]}, N_ar<={k[1]}): {n} ev" for k, n in sorted(shape_hist.items()))
    )

    stats = {
        "gpu_wall": 0.0,
        "post_wall": 0.0,
        "post_drain_wall": 0.0,
        "fallback": 0,
        "fallback_wall": 0.0,
        "batched": 0,
        "compiles": 0,
        "program_compiles": 0,
        "gpu_mib_max": 0.0,
        "buckets": [],
    }
    inflight: list[tuple[int, list[dict]]] = []
    worker_ok = {"alive": True}

    def post_task(prep: dict, result_obj, reason: str | None) -> tuple:
        """Picklable payload for ONE post-processing job.

        The prep dict is shallow-copied so the driver can drop its own
        ``prep["data"]`` the moment the job is handed over: the copy keeps the
        event data alive until the executor has pickled it, and mutating the
        driver's dict afterwards cannot reach the worker.
        """
        return make_post_task(dict(prep), result_obj, reason)

    def post_info(prep: dict, result_obj) -> dict:
        """The little the driver has to remember about an outstanding job."""
        return {
            "event_id": prep["event_id"],
            "dat_path": prep["dat_path"],
            "log_path": prep.get("log_path") or event_log_path(event_log_dir, prep["event_id"]),
            "path": "gpu-batched" if result_obj is not None else "cpu-fallback",
        }

    def integrate_post(info: dict, report: dict | None, error) -> None:
        """Fold ONE finished post-processing job into results + progress line."""
        if error is None:
            stats["batched"] += int(report.get("batched", 0))
            stats["fallback"] += int(report.get("fallback", 0))
            stats["fallback_wall"] += float(report.get("fallback_wall", 0.0))
            outcome = report["outcome"]
        else:
            # The worker raised or died before it could report: this event is
            # failed, but it still gets exactly one entry in results.
            tb = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            print(f"  post-processing worker failed for {info['event_id']}:\n{tb}")
            outcome = {
                "event_id": info["event_id"],
                "dat_path": info["dat_path"],
                "ok": False,
                "error": str(error),
                "traceback": tb,
                "path": info["path"],
                "log_path": info["log_path"],
            }
        results.append(outcome)
        status = outcome.get("status", "done" if outcome["ok"] else "failed")
        print(
            f"[{len(results)}/{total}] {status}: {outcome['event_id']} "
            f"[{outcome.get('path', 'gpu-batched')}] log={outcome.get('log_path')}",
            flush=True,
        )

    def finalize_one(prep: dict, result_obj, reason: str | None) -> None:
        """Post-process one event HERE, with its console output in its own log."""
        task = post_task(prep, result_obj, reason)
        info = post_info(prep, result_obj)
        try:
            report, error = post_task_fn(task), None
        except Exception as exc:  # noqa: BLE001 - one event fails, run goes on
            report, error = None, exc
        integrate_post(info, report, error)

    def process_chunk(chunk: list[dict], sampled: dict, failures: dict, chunk_error) -> None:
        # With a pool this only times submission (which blocks while the
        # outstanding bound is full), i.e. the part that is still on the GPU
        # critical path; the work itself happens in the children.
        t_post = time.time()
        for prep in chunk:
            event_id = prep["event_id"]
            result_obj = sampled.get(event_id)
            reason = None
            if result_obj is None:
                reason = (
                    chunk_error
                    or failures.get(event_id)
                    or "the batched sampler returned no result for this event"
                )
            if post_pool is None:
                finalize_one(prep, result_obj, reason)
            else:
                post_pool.submit(post_task(prep, result_obj, reason), post_info(prep, result_obj))
            prep["data"] = None  # release the event data as soon as it is handed over
        stats["post_wall"] += time.time() - t_post
        gc.collect()

    # The pool has to be created after pin_this_process_to_cpu above: its
    # children inherit JAX_PLATFORMS=cpu from this process, and only the batched
    # sampler worker is allowed on the GPU. --post-workers 0 keeps the old
    # behaviour, where process_chunk finalizes every event inline.
    post_pool = None
    if post_workers > 0:
        post_pool = PostProcessPool(
            post_workers,
            integrate_post,
            post_task_fn,
            label="Step 4 post-processing pool",
            job_timeout=post_job_timeout,
            poll_seconds=post_poll_seconds,
            max_tasks_per_child=post_tasks_per_child,
        )
        print(
            f"Post-processing pool: {post_workers} worker process(es), at most "
            f"{post_pool.max_pending} outstanding job(s) before the driver blocks."
        )
    else:
        print("Post-processing: serial in the driver (--post-workers 0).")

    def drain_one() -> None:
        chunk_id, chunk = inflight.pop(0)
        if not worker_ok["alive"]:
            process_chunk(chunk, {}, {}, "the fp32 GPU worker is gone")
            return
        budget = max(
            chunk_timeout_min, chunk_timeout_per_event * max(1, len(chunk))
        )
        try:
            message = worker.collect(timeout=budget)
        except RuntimeError as exc:
            print(
                f"ERROR: submission {chunk_id} ({len(chunk)} events) did not come back: "
                f"{exc}. Killing the worker; this and every remaining event goes to the "
                "unbatched CPU path."
            )
            worker_ok["alive"] = False
            worker.kill()
            process_chunk(chunk, {}, {}, f"fp32 GPU worker lost: {exc}")
            return
        if message.get("chunk_id") != chunk_id:
            raise RuntimeError(
                f"GPU worker returned chunk {message.get('chunk_id')}, expected {chunk_id}"
            )
        stats["gpu_wall"] += float(message.get("wall", 0.0))
        stats["compiles"] += int(message.get("compiles") or 0)
        stats["program_compiles"] += int(message.get("program_compiles") or 0)
        stats["buckets"].extend(message.get("buckets", []))
        if message.get("gpu_mib"):
            stats["gpu_mib_max"] = max(stats["gpu_mib_max"], float(message["gpu_mib"]))
        process_chunk(
            chunk,
            message.get("results") or {},
            message.get("failures") or {},
            message.get("error") or message.get("unsupported"),
        )

    # Ctrl-C: stop submitting, drain what is in flight, then shut down cleanly.
    interrupted = {"flag": False}
    terminated = {"flag": False}
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_sigint(signum, frame):  # noqa: ANN001
        if interrupted["flag"]:  # second Ctrl-C: give up now
            signal.signal(signal.SIGINT, previous_sigint)
            raise KeyboardInterrupt
        interrupted["flag"] = True
        print(
            "\ninterrupt received: no new submissions; finishing the "
            f"{len(inflight)} submission(s) already on the GPU. Ctrl-C again to abort.",
            flush=True,
        )

    def _on_sigterm(signum, frame):  # noqa: ANN001
        # A scheduler's SIGTERM is followed by SIGKILL: there is no time to
        # drain. Unwind now so the finally blocks kill the post-processing
        # children (which would otherwise be re-parented to init while writing
        # into the output tree) and the fp32 GPU worker (which holds the A40).
        terminated["flag"] = True
        interrupted["flag"] = True
        print("\nSIGTERM received: shutting down and killing the workers.", flush=True)
        raise KeyboardInterrupt("SIGTERM")

    t_sample = time.time()
    submitted_events: list[str] = []
    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:  # not the main thread
        previous_sigint = None
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:  # not the main thread
        previous_sigterm = None
    try:
        for chunk_id, chunk in enumerate(chunks):
            if interrupted["flag"]:
                break
            worker.submit(chunk_id, [(p["event_id"], p["data"]) for p in chunk])
            submitted_events.extend(p["event_id"] for p in chunk)
            inflight.append((chunk_id, chunk))
            # Two slots: while the worker samples submission i, the driver
            # post-processes submission i-1 (Q metrics, CSV, pickles, context).
            if len(inflight) > 1:
                drain_one()
        while inflight:
            drain_one()
    finally:
        if previous_sigint is not None:
            try:
                signal.signal(signal.SIGINT, previous_sigint)
            except ValueError:
                pass
        if previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
            except ValueError:
                pass
        if post_pool is not None:
            # Nothing may still be in flight when the summary prints: drain on
            # the way out of an exception too, or events would be lost. The
            # drain is bounded (a pool that produces nothing for
            # post_job_timeout is torn down), and skipped entirely on SIGTERM,
            # where the scheduler's SIGKILL is already on its way.
            t_drain = time.time()
            try:
                if terminated["flag"]:
                    post_pool.abort("driver received SIGTERM")
                else:
                    post_pool.close()
            finally:
                stats["post_drain_wall"] = time.time() - t_drain
                stats["post_wall"] += stats["post_drain_wall"]
    sample_wall = time.time() - t_sample

    if interrupted["flag"]:
        never_run = [p["event_id"] for p in preps if p["event_id"] not in set(submitted_events)]
        print(
            f"\nINTERRUPTED: {len(never_run)} loaded event(s) were never submitted and "
            "have no output directory: " + (", ".join(never_run) if never_run else "(none)")
        )

    print(
        f"\nGPU batched summary: {stats['batched']} events sampled on the GPU, "
        f"{stats['fallback']} fell back to the unbatched CPU path"
        + (f" ({stats['fallback_wall']:.1f} s)." if stats["fallback"] else ".")
    )
    print(
        f"  load {load_wall:.1f} s | sample+post {sample_wall:.1f} s "
        f"(GPU busy {stats['gpu_wall']:.1f} s, driver blocked on post-processing "
        f"{stats['post_wall']:.1f} s) | XLA compilations {stats['compiles']} total, "
        f"{stats['program_compiles']} of them batched programs "
        f"(up to {len(GPU_PROGRAM_NAMES)} per distinct shape/width)"
    )
    if post_pool is not None:
        print(
            f"  post-processing pool: {post_workers} worker(s), {post_pool.submitted} job(s), "
            f"peak {post_pool.max_outstanding}/{post_pool.max_pending} outstanding, "
            f"{post_pool.inline} run in the driver, {post_pool.isolated} in a one-shot "
            f"subprocess, {post_pool.expired} timed out, {post_pool.restarts} pool "
            f"restart(s); final drain {stats['post_drain_wall']:.1f} s"
        )
    else:
        print("  post-processing pool: disabled; every event was finalized in the driver.")
    if not interrupted["flag"] and len(results) != total:
        print(
            f"  WARNING: {len(results)} result(s) for {total} event(s); every event "
            "should appear exactly once."
        )
    wall = load_wall + sample_wall
    if wall > 0 and total:
        print(
            f"  throughput: {total / wall * 3600.0:.0f} events/hour end-to-end "
            f"({stats['batched'] / stats['gpu_wall'] * 3600.0:.0f} ev/h sampler-only)"
            if stats["gpu_wall"] > 0
            else f"  throughput: {total / wall * 3600.0:.0f} events/hour end-to-end"
        )
    if stats["gpu_mib_max"]:
        print(f"  peak device memory seen at chunk boundaries: {stats['gpu_mib_max']:.0f} MiB")
    if stats["buckets"]:
        print("  buckets: " + "; ".join(stats["buckets"]))
    return results


def run_event_batch_cpu(
    run_task_fn,
    tasks,
    *,
    total: int,
    event_workers: int,
    max_events_per_worker: int,
) -> list[dict]:
    """Run prepared event tasks serially or across a bounded CPU worker pool.

    ``run_task_fn`` is the picklable, module-level function that inverts one
    event from its task tuple and returns the per-event result dict.  Device
    resolution, sampler-path provenance and per-event log directories are the
    caller's business; this only executes.
    """
    if event_workers == 1:
        results = []
        for task in tasks:
            result = run_task_fn(task)
            results.append(result)
            if not result["ok"]:
                print(f"Warning: Inversion execution failed for {result['dat_path']}: {result['error']}")
        return results

    print(f"Running {total} events across {event_workers} event worker processes.")

    results = []
    ctx = mp.get_context("spawn")
    result_iter = bounded_process_pool_map(
        run_task_fn,
        tasks,
        max_workers=event_workers,
        mp_context=ctx,
        max_tasks_per_child=max_events_per_worker,
        label="Step 4 event worker pool",
    )
    for done_count, result in enumerate(result_iter, 1):
        results.append(result)
        status = result.get("status", "failed" if not result["ok"] else "done")
        log_note = f" log={result['log_path']}" if result.get("log_path") else ""
        print(f"[{done_count}/{total}] {status}: {result['event_id']}{log_note}", flush=True)
    return results

"""
Bounded ProcessPoolExecutor helpers.

The stdlib executor does not limit the number of submitted futures.  For large
event/station batches, submitting everything at once can retain substantial
input and result data in the parent process.  This helper keeps only a small
window of futures active while still yielding results as workers finish.
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Iterable, Iterator, TypeVar

from .process_cleanup import managed_process_pool

T = TypeVar("T")
R = TypeVar("R")


def bounded_process_pool_map(
    func: Callable[[T], R],
    tasks: Iterable[T],
    *,
    max_workers: int,
    mp_context,
    initializer: Callable[[], None] | None = None,
    max_tasks_per_child: int | None = None,
    pending_multiplier: int = 2,
    label: str = "process pool",
) -> Iterator[R]:
    """Yield results from a bounded ProcessPoolExecutor task stream."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_tasks_per_child is not None and max_tasks_per_child < 1:
        raise ValueError("max_tasks_per_child must be >= 1")
    if pending_multiplier < 1:
        raise ValueError("pending_multiplier must be >= 1")

    executor_kwargs = {
        "max_workers": max_workers,
        "mp_context": mp_context,
    }
    if initializer is not None:
        executor_kwargs["initializer"] = initializer
    if max_tasks_per_child is not None:
        # ProcessPoolExecutor gained max_tasks_per_child in Python 3.11;
        # on older interpreters run without worker recycling.
        if sys.version_info >= (3, 11):
            executor_kwargs["max_tasks_per_child"] = max_tasks_per_child
        else:
            warnings.warn(
                f"{label}: max_tasks_per_child requires Python >= 3.11 "
                f"(running {sys.version_info.major}.{sys.version_info.minor}); "
                "workers will not be recycled.",
                RuntimeWarning,
                stacklevel=2,
            )

    max_pending = max_workers * pending_multiplier
    task_iter = iter(tasks)
    pending = set()

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        with managed_process_pool(executor, label=label):
            for _ in range(max_pending):
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(func, task))

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()

                for _ in range(max_pending - len(pending)):
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    pending.add(executor.submit(func, task))

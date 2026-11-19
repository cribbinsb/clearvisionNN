"""Running a stage's per-item work on several threads.

The heavy lifting is in ffmpeg subprocesses and in numpy and OpenCV calls that
release the interpreter lock, so threads are enough.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from clearvisionnn.pipeline.records import Failure

A = TypeVar("A")
B = TypeVar("B")

_PROGRESS_EVERY = 50


@dataclass
class Outcome(Generic[B]):
    """What a stage's items produced, and which items failed."""

    results: list[B] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)


def map_items(
    name: str,
    function: Callable[[A], B],
    items: Sequence[A],
    workers: int,
    describe: Callable[[A], str],
    tolerated: tuple[type[Exception], ...] = (),
    checkpoint: Callable[[], None] | None = None,
) -> Outcome[B]:
    """`function` over `items`, in order, reporting progress under the stage `name`.

    Only errors of the `tolerated` types (an unreadable source file, say) are
    recorded as failures and skipped; anything else is a fault in the
    configuration, the code or the machine, and stops the stage. `checkpoint`
    runs between batches of items, for periodic checks such as the disk budget.
    """
    outcome: Outcome[B] = Outcome()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), _PROGRESS_EVERY):
            batch = items[start : start + _PROGRESS_EVERY]
            if checkpoint is not None:
                checkpoint()
            for item, future in [(item, pool.submit(function, item)) for item in batch]:
                try:
                    outcome.results.append(future.result())
                except tolerated as error:
                    outcome.failures.append(Failure(item=describe(item), error=str(error)[-500:]))
            done = start + len(batch)
            print(f"[{name}] {done}/{len(items)} ({len(outcome.failures)} failed)", flush=True)
    return outcome

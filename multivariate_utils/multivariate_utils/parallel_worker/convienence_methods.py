from typing import Callable, List, Optional, Any
from .resource_checker import ResourceMonitor, ResourceInfo
from .executer import SmartParallelExecutor

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def smart_map(
    func: Callable,
    items,
    mode: str = "auto",
    max_workers: Optional[int] = None,
    show_progress: bool = False,
    verbose: bool = True,
) -> List[Any]:
    """
    One-liner parallel map.  Picks the right strategy, monitors
    resources, and falls back cleanly.

    Returns the results list directly.  Use SmartParallelExecutor.map()
    if you need the full ExecutionResult (error list, timing, etc.).
    """
    executor = SmartParallelExecutor(
        mode=mode, max_workers=max_workers, verbose=verbose
    )
    result = executor.map(func, items, show_progress=show_progress)

    if verbose:
        print(result)

    return result.results


def smart_filter(
    predicate: Callable[[Any], bool],
    items,
    mode: str = "thread",
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Parallel filter.  Returns items for which predicate returned True.
    """
    # FIX 10: materialise once so smart_map and the zip below both
    # see the same concrete list — not a one-shot generator.
    items = list(items)
    booleans = smart_map(
        predicate, items, mode=mode, max_workers=max_workers, verbose=False
    )
    return [item for item, keep in zip(items, booleans) if keep]


# ============================================================================
# UTILITY
# ============================================================================


def get_system_resources() -> ResourceInfo:
    return ResourceMonitor.get()


def estimate_optimal_workers(
    mode: str = "thread", task_memory_mb: Optional[float] = None
) -> int:
    return ResourceMonitor.optimal_workers(mode, task_memory_mb)

import time
import logging
import traceback
import pickle
from typing import Callable, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

from .resource_checker import ResourceMonitor
from .result_resources import ExecutionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# SMART PARALLEL EXECUTOR
# ============================================================================


class SmartParallelExecutor:
    """
    Parallel executor with resource awareness and clean fallback.

    Design rules:
      - Every error is printed and logged at WARNING.  Nothing swallowed.
      - On memory breach mid-run, only unfinished items are retried
        sequentially.  Completed results are kept.
      - Accepts any iterable (list, range, generator, ...).
      - Reusable: all state is local to each map() call.
    """

    def __init__(
        self,
        mode: str = "auto",
        max_workers: Optional[int] = None,
        task_memory_mb: Optional[float] = None,
        enable_fallback: bool = True,
        memory_check_interval: int = 100,
        verbose: bool = True,
    ):
        self.mode = mode
        self.max_workers = max_workers
        self.task_memory_mb = task_memory_mb
        self.enable_fallback = enable_fallback
        self.memory_check_interval = max(1, memory_check_interval)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def map(
        self, func: Callable, items, show_progress: bool = False
    ) -> ExecutionResult:
        """
        Apply func to every item.  Picks execution strategy, monitors
        resources, falls back cleanly, and reports every error.
        """
        # FIX 3: materialise input once — works with generators, ranges, anything
        # FIX 6: no instance state carried between calls (everything is local)
        try:
            items = list(items)
        except TypeError as exc:
            raise TypeError(
                f"'items' must be iterable (got {type(items).__name__})"
            ) from exc

        if not items:
            return ExecutionResult(
                results=[],
                success_count=0,
                failure_count=0,
                execution_time=0.0,
                mode_used="none",
                workers_used=0,
                fell_back=False,
                fallback_reason="",
                errors=[],
            )

        start = time.time()

        # ── pre-flight: can we even parallelize? ─────────────────────
        ok, reason = ResourceMonitor.can_parallelize()
        if not ok:
            if self.verbose:
                print(f"\n⚠️  Pre-flight: {reason} — running sequentially.")
            logger.warning("Pre-flight failed: %s", reason)
            return self._run_sequential(
                func, items, start, fell_back=True, reason=reason
            )

        # ── resolve mode ─────────────────────────────────────────────
        mode = (
            self.mode if self.mode in ("thread", "process", "sequential") else "thread"
        )

        # FIX 4: process mode pre-check — is func picklable?
        if mode == "process":
            try:
                pickle.dumps(func)
            except Exception as exc:
                msg = (
                    f"'{getattr(func, '__name__', repr(func))}' can't be pickled "
                    f"({type(exc).__name__}: {exc}). "
                    f"Lambdas and nested functions don't work with multiprocessing."
                )
                logger.warning(msg)
                if self.enable_fallback:
                    if self.verbose:
                        print(
                            f"\n⚠️  Pickle pre-check failed — falling back to thread mode."
                        )
                        print(f"   {msg}")
                    mode = "thread"
                else:
                    raise TypeError(msg) from exc

        # ── calculate workers ────────────────────────────────────────
        workers = ResourceMonitor.optimal_workers(
            mode, self.task_memory_mb, self.max_workers
        )

        if self.verbose:
            print(f"\n📊 {ResourceMonitor.get()}")
            print(f"🚀 mode={mode}  workers={workers}  items={len(items)}")

        # ── execute ──────────────────────────────────────────────────
        try:
            if mode == "thread":
                return self._run_pool(
                    ThreadPoolExecutor,
                    "thread",
                    func,
                    items,
                    workers,
                    start,
                    show_progress,
                )
            elif mode == "process":
                return self._run_pool(
                    ProcessPoolExecutor,
                    "process",
                    func,
                    items,
                    workers,
                    start,
                    show_progress,
                )
            else:
                return self._run_sequential(func, items, start)

        except Exception as exc:
            # FIX 11: print full traceback so the user can find the source
            tb = traceback.format_exc()
            logger.error("Executor-level failure:\n%s", tb)

            if self.enable_fallback:
                reason = f"{type(exc).__name__}: {exc}"
                if self.verbose:
                    print(f"\n❌ Executor crashed: {reason}")
                    print(f"   Falling back to sequential for all {len(items)} items.")
                return self._run_sequential(
                    func, items, start, fell_back=True, reason=reason
                )
            raise

    # ------------------------------------------------------------------
    # POOL EXECUTION  (thread and process share one code path)
    # ------------------------------------------------------------------

    def _run_pool(
        self, PoolClass, pool_name, func, items, workers, start, show_progress
    ) -> ExecutionResult:
        """
        Submit all items to the pool, collect results, and handle a
        mid-run memory breach by filling in only the missing items
        sequentially afterward — completed results are never discarded.
        """
        completed = {}  # idx -> result value  (includes legitimate None returns)
        failed = set()  # indices whose tasks raised an exception
        errors = []
        breach_msg = None  # set to a reason string if memory limit is hit

        pbar = self._progress_bar(len(items)) if show_progress else None

        with PoolClass(max_workers=workers) as pool:
            future_map = {
                pool.submit(func, item): idx for idx, item in enumerate(items)
            }

            for i, future in enumerate(as_completed(future_map)):
                idx = future_map[future]

                # FIX 8: skip i == 0 — nothing has run yet, snapshot is meaningless
                if i > 0 and i % self.memory_check_interval == 0:
                    res = ResourceMonitor.get()
                    if res.memory_percent > 95:
                        breach_msg = (
                            f"memory hit {res.memory_percent:.1f}% after {i} items"
                        )
                        logger.warning("Memory breach: %s", breach_msg)
                        # Cancel what we can (only pending futures honour this).
                        for f in future_map:
                            f.cancel()
                        # FIX 7: break cleanly instead of raising inside with-block.
                        # The with-block calls shutdown(wait=True), letting
                        # in-flight futures finish.  We collect their results below.
                        break

                # ── collect this future ──
                try:
                    completed[idx] = future.result()
                except Exception as exc:
                    # FIX 2: log at WARNING (visible at default level) + print
                    err = f"Item {idx}: {type(exc).__name__}: {exc}"
                    errors.append(err)
                    completed[idx] = None
                    failed.add(idx)
                    logger.warning(err)
                    if self.verbose:
                        print(f"   ⚠️  {err}")

                if pbar:
                    pbar.update(1)

        # with-block exited — pool is shut down, all in-flight futures are done.

        if pbar:
            pbar.close()

        # ── after a breach, collect results from futures that finished
        #    while shutdown was waiting (we broke before reading them)
        if breach_msg:
            for future, idx in future_map.items():
                if idx not in completed and future.done() and not future.cancelled():
                    try:
                        completed[idx] = future.result()
                    except Exception as exc:
                        err = f"Item {idx}: {type(exc).__name__}: {exc}"
                        errors.append(err)
                        completed[idx] = None
                        failed.add(idx)
                        logger.warning(err)
                        if self.verbose:
                            print(f"   ⚠️  {err}")

        # ── FIX 7 cont.: fallback runs ONLY items that never executed ──
        fell_back = False
        fallback_reason = ""

        if breach_msg:
            if self.enable_fallback:
                fell_back = True
                fallback_reason = breach_msg
                missing = [i for i in range(len(items)) if i not in completed]

                if self.verbose:
                    print(f"\n⚠️  {breach_msg}")
                    print(f"   {len(completed)} of {len(items)} finished in parallel.")
                    print(f"   Running {len(missing)} remaining items sequentially …")

                for idx in missing:
                    try:
                        completed[idx] = func(items[idx])
                    except Exception as exc:
                        err = f"Item {idx} (sequential retry): {type(exc).__name__}: {exc}"
                        errors.append(err)
                        completed[idx] = None
                        failed.add(idx)
                        logger.warning(err)
                        if self.verbose:
                            print(f"   ⚠️  {err}")
            else:
                raise MemoryError(breach_msg)

        # ── assemble ordered results ─────────────────────────────────
        final = [completed.get(i) for i in range(len(items))]
        mode_label = pool_name if not fell_back else f"{pool_name}→sequential"

        return ExecutionResult(
            results=final,
            success_count=len(items) - len(failed),
            failure_count=len(failed),
            execution_time=time.time() - start,
            mode_used=mode_label,
            workers_used=workers,
            fell_back=fell_back,
            fallback_reason=fallback_reason,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # SEQUENTIAL  (standalone or as fallback)
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        func,
        items,
        start,
        fell_back=False,
        reason="",
    ) -> ExecutionResult:
        results = []
        errors = []
        failed = set()

        # FIX 5: print MORE when falling back, not less
        if self.verbose:
            if fell_back:
                logger.info(
                    f"Running {len(items)} items sequentially (reason: {reason})"
                )
            else:
                logger.info(f"Sequential mode: {len(items)} items")
        for idx, item in enumerate(items):
            try:

                # logging each item start in verbose mode
                if self.verbose:
                    logger.info(f"Processing item {idx + 1}/{len(items)}")
                results.append(func(item))
            except Exception as exc:
                # FIX 2: WARNING + print
                err = f"Item {idx}: {type(exc).__name__}: {exc}"
                errors.append(err)
                results.append(None)
                failed.add(idx)
                logger.warning(err)
                if self.verbose:
                    print(f"   ⚠️  {err}")

        return ExecutionResult(
            results=results,
            success_count=len(items) - len(failed),
            failure_count=len(failed),
            execution_time=time.time() - start,
            mode_used="sequential",
            workers_used=1,
            fell_back=fell_back,
            fallback_reason=reason,
            errors=errors,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _progress_bar(total):
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc="Processing", unit="item")
        except ImportError:
            return None

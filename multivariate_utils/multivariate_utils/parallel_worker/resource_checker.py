from dataclasses import dataclass
import psutil
from typing import Optional, Tuple
import multiprocessing as mp


@dataclass
class ResourceInfo:
    cpu_count: int
    memory_total_gb: float
    memory_available_gb: float
    memory_percent: float
    cpu_percent: float

    def __str__(self) -> str:
        return (
            f"CPU: {self.cpu_count} cores ({self.cpu_percent:.1f}% used) | "
            f"RAM: {self.memory_available_gb:.1f} GB free / "
            f"{self.memory_total_gb:.1f} GB total ({self.memory_percent:.1f}% used)"
        )

    def is_healthy(self, mem_thresh: float = 80.0, cpu_thresh: float = 90.0) -> bool:
        return self.memory_percent < mem_thresh and self.cpu_percent < cpu_thresh


# ============================================================================
# RESOURCE MONITOR
# ============================================================================


class ResourceMonitor:

    @staticmethod
    def get() -> ResourceInfo:
        mem = psutil.virtual_memory()
        return ResourceInfo(
            cpu_count=mp.cpu_count() or 1,
            memory_total_gb=mem.total / 1024**3,
            memory_available_gb=mem.available / 1024**3,
            memory_percent=mem.percent,
            cpu_percent=psutil.cpu_percent(interval=0.1),
        )

    @staticmethod
    def optimal_workers(
        mode: str,
        task_mb: Optional[float] = None,
        cap: Optional[int] = None,
    ) -> int:
        res = ResourceMonitor.get()
        base = min(32, res.cpu_count * 4) if mode == "thread" else res.cpu_count

        if task_mb and task_mb > 0:
            avail_mb = res.memory_available_gb * 1024 * 0.8
            base = min(base, max(1, int(avail_mb / task_mb)))

        if cap:
            base = min(base, cap)

        return max(1, base)

    @staticmethod
    def can_parallelize(min_mem_gb: float = 0.5) -> Tuple[bool, str]:
        res = ResourceMonitor.get()
        if res.memory_available_gb < min_mem_gb:
            return (
                False,
                f"only {res.memory_available_gb:.2f} GB free (need >= {min_mem_gb} GB)",
            )
        if res.memory_percent > 97:
            return False, f"memory already at {res.memory_percent:.1f}%"
        if res.cpu_percent > 98:
            return False, f"CPU already at {res.cpu_percent:.1f}%"
        return True, "OK"

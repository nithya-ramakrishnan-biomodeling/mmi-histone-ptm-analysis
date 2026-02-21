from enum import Enum
from typing import List, Any
from dataclasses import dataclass, field


# ============================================================================
# ENUMS AND CONFIGS
# ============================================================================


class ExecutionMode(Enum):
    """Execution modes."""

    THREAD = "thread"
    PROCESS = "process"
    SEQUENTIAL = "sequential"
    AUTO = "auto"  # Automatically choose best mode


@dataclass
class ExecutionResult:
    results: List[Any]
    success_count: int
    failure_count: int
    execution_time: float
    mode_used: str
    workers_used: int
    fell_back: bool
    fallback_reason: str
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        total = self.success_count + self.failure_count
        status = "✅" if self.failure_count == 0 else "⚠️"

        # FIX 9: guard against ZeroDivisionError when execution is instant
        if self.execution_time > 0:
            throughput = f"{total / self.execution_time:.0f} items/s"
        else:
            throughput = "N/A"

        lines = [
            "",
            f"{status} Execution Complete",
            f"   Mode       : {self.mode_used} ({self.workers_used} worker{'s' if self.workers_used != 1 else ''})",
            f"   Results    : {self.success_count} ok  /  {self.failure_count} failed  (total {total})",
            f"   Time       : {self.execution_time:.4f}s   Throughput: {throughput}",
        ]

        if self.fell_back:
            lines.append(f"   Fell back  : {self.fallback_reason}")

        # FIX 1: actually print the error messages, not just the count
        if self.errors:
            lines.append(f"   Errors ({len(self.errors)}):")
            for e in self.errors[:5]:
                lines.append(f"      → {e}")
            if len(self.errors) > 5:
                lines.append(f"      … and {len(self.errors) - 5} more.")

        return "\n".join(lines)

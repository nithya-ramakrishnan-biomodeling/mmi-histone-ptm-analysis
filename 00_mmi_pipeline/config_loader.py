import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    name: str
    enabled: bool = True
    args: dict = field(default_factory=dict)

    def to_cli_args(self) -> list[str]:
        """Convert the args dict to a flat list of CLI arguments."""
        cli_args = []
        for key, value in self.args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cli_args.append(str(key))
            else:
                cli_args.append(str(key))
                cli_args.append(str(value))
        return cli_args


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    stop_on_error: bool = True
    variables: dict[str, str] = field(default_factory=dict)
    steps: dict[str, StepConfig] = field(default_factory=dict)


def _resolve_variables(value, variables: dict[str, str], max_depth: int = 10) -> str:
    """
    Resolve ${variable_name} references in a string value.
    Supports nested references up to max_depth levels.
    """
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{(\w+)\}")

    for _ in range(max_depth):
        matches = pattern.findall(value)
        if not matches:
            break
        for var_name in matches:
            if var_name in variables:
                value = value.replace(f"${{{var_name}}}", str(variables[var_name]))
            else:
                raise ValueError(
                    f"Undefined variable '${{{var_name}}}'. "
                    f"Available variables: {list(variables.keys())}"
                )
    return value


def _resolve_dict(d: dict, variables: dict[str, str]) -> dict:
    """Recursively resolve all variable references in a dictionary."""
    resolved = {}
    for key, value in d.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_dict(value, variables)
        elif isinstance(value, list):
            resolved[key] = [_resolve_variables(v, variables) for v in value]
        elif isinstance(value, str):
            resolved[key] = _resolve_variables(value, variables)
        else:
            resolved[key] = value
    return resolved


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file with variable resolution."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # ── Step 1: Parse variables ──
    variables = raw.get("variables", {})

    # Resolve variables that reference other variables (e.g., raw_data uses data_dir)
    # Iterate multiple times to handle chained references
    for _ in range(10):
        changed = False
        for key, value in variables.items():
            if isinstance(value, str) and "${" in value:
                resolved = _resolve_variables(value, variables)
                if resolved != value:
                    variables[key] = resolved
                    changed = True
        if not changed:
            break

    # ── Step 2: Parse global settings ──
    stop_on_error = raw.get("stop_on_error", True)

    # ── Step 3: Parse and resolve per-step configs ──
    steps = {}
    raw_steps = raw.get("steps", {})
    for step_name, step_data in raw_steps.items():
        if step_data is None:
            step_data = {}

        # Resolve variable references in step args
        raw_args = step_data.get("args", {})
        resolved_args = _resolve_dict(raw_args, variables) if raw_args else {}

        steps[step_name] = StepConfig(
            name=step_name,
            enabled=step_data.get("enabled", True),
            args=resolved_args,
        )

    config = PipelineConfig(
        stop_on_error=stop_on_error,
        variables=variables,
        steps=steps,
    )

    return config
import click
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from multivariate_utils import log_header
from config_loader import load_pipeline_config


PIPELINE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = PIPELINE_DIR.joinpath("00_mmi_pipeline", "pipeline_config.yaml")


@dataclass
class PipelineStep:
    """Represents a single pipeline step."""
    name: str
    script: str
    description: str
    cli_options: list[click.Option] = field(default_factory=list)


# ──────────────────────────────────────────────
# Register all pipeline steps here
# ──────────────────────────────────────────────
PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        name="preprocess",
        script="00_data_preprocessing.py",
        description="Data preprocessing: normalization, imputation, clipping",
    ),
    PipelineStep(
        name="mmi",
        script="01_mmi_calculation.py",
        description="Multivariate Mutual Information analysis",
    ),
    PipelineStep(
        name="feature",
        script="02_feature_selection.py",
        description="Feature selection based on MMI results",
    ),
    PipelineStep(
        name="prediction",
        script="03_regression_prediction.py",
        description="Predictive modeling based on selected features",
    ),
]

STEP_MAP: dict[str, PipelineStep] = {s.name: s for s in PIPELINE_STEPS}


def _run_script(step: PipelineStep, extra_args: tuple[str, ...] | list[str]) -> int:
    """Run a pipeline step script as a subprocess."""
    script_path = PIPELINE_DIR.joinpath("00_mmi_pipeline", step.script)
    if not script_path.exists():
        click.echo(f"Error: Script '{step.script}' not found at {script_path}")
        return 1

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  Step: {step.name}")
    click.echo(f"  Script: {step.script}")
    click.echo(f"  Description: {step.description}")
    if extra_args:
        click.echo(f"  Args: {' '.join(extra_args)}")
    click.echo(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(script_path), *extra_args],
        cwd=str(PIPELINE_DIR),
    )
    return result.returncode


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
@click.group()
def cli():
    """MMI Pipeline Runner — run individual steps or the full pipeline."""
    pass


@cli.command(name="list")
def list_steps():
    """List all registered pipeline steps."""
    if not PIPELINE_STEPS:
        click.echo("No pipeline steps registered.")
        return
    click.echo("Registered pipeline steps:\n")
    click.echo(f"  {'#':<4} {'Name':<25} {'Script':<35} {'Description'}")
    click.echo(f"  {'─'*4} {'─'*25} {'─'*35} {'─'*30}")
    for i, step in enumerate(PIPELINE_STEPS):
        click.echo(f"  {i:<4} {step.name:<25} {step.script:<35} {step.description}")


@cli.command(name="run",
             context_settings=dict(
                 ignore_unknown_options=True,
                 allow_extra_args=True,
             ))
@click.argument("step_name", type=click.Choice(list(STEP_MAP.keys()), case_sensitive=False))
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Path to YAML config file. Overrides CLI args for this step.")
@click.argument("step_args", nargs=-1, type=click.UNPROCESSED)
def run_step(step_name, config, step_args):
    """
    Run a single pipeline step by name.

    \b
    Use config file:
        python mmi_main.py run preprocess -c pipeline_config.yaml

    Or pass CLI args directly after '--':
        python mmi_main.py run preprocess -- -d data.csv -n z_score --impute-missing-values
    """
    step = STEP_MAP[step_name]

    # If config file provided, load args from it
    if config:
        pipeline_config = load_pipeline_config(config)
        step_config = pipeline_config.steps.get(step_name)
        if step_config is None:
            click.echo(f"Warning: Step '{step_name}' not found in config. Running with no args.")
            final_args = list(step_args)
        elif not step_config.enabled:
            click.echo(f"Step '{step_name}' is disabled in config. Skipping.")
            return
        else:
            # Config args + any extra CLI args (CLI overrides)
            final_args = step_config.to_cli_args() + list(step_args)
    else:
        final_args = list(step_args)

    returncode = _run_script(step, final_args)

    if returncode != 0:
        click.echo(f"\nStep '{step.name}' FAILED (exit code {returncode}).")
        sys.exit(returncode)
    else:
        click.echo(f"\nStep '{step.name}' completed successfully.")


@cli.command(name="run-all")
@click.option("--config", "-c", type=click.Path(exists=True), default=str(DEFAULT_CONFIG),
              help="Path to YAML config file with per-step arguments.")
@click.option("--stop-on-error/--no-stop-on-error", default=None,
              help="Stop pipeline if a step fails. Overrides config setting.")
@click.option("--step-args", "-a", multiple=True, type=str,
              help="Override per-step arguments: 'STEP_NAME:ARGS'. Takes precedence over config.")
def run_all(config, stop_on_error, step_args):
    """
    Run all pipeline steps in registered order using a config file.

    \b
    Examples:
        python mmi_main.py run-all
        python mmi_main.py run-all -c my_config.yaml
        python mmi_main.py run-all -c pipeline_config.yaml -a "preprocess:--clip-min-value -5"
    """
    if not PIPELINE_STEPS:
        click.echo("No pipeline steps registered.")
        return

    # Load config
    config_path = Path(config)
    if config_path.exists():
        pipeline_config = load_pipeline_config(config_path)
        click.echo(f"Loaded config from: {config_path}")
    else:
        click.echo(f"Warning: Config file '{config_path}' not found. Running with no args.")
        from config_loader import PipelineConfig
        pipeline_config = PipelineConfig()

    # Determine stop_on_error (CLI flag overrides config)
    effective_stop = stop_on_error if stop_on_error is not None else pipeline_config.stop_on_error

    # Parse CLI override args
    cli_override_map: dict[str, list[str]] = {}
    for arg_entry in step_args:
        if ":" not in arg_entry:
            click.echo(f"Warning: Ignoring malformed step-arg '{arg_entry}'. "
                       f"Expected format 'STEP_NAME:ARGS'.")
            continue
        name, args_str = arg_entry.split(":", 1)
        name = name.strip()
        if name not in STEP_MAP:
            click.echo(f"Warning: Unknown step name '{name}'. Skipping.")
            click.echo(f"  Available: {', '.join(STEP_MAP.keys())}")
            continue
        cli_override_map[name] = args_str.split()

    click.echo(f"Running {len(PIPELINE_STEPS)} pipeline step(s)...\n")

    passed = 0
    failed = 0
    skipped = 0

    for i, step in enumerate(PIPELINE_STEPS):
        step_config = pipeline_config.steps.get(step.name)

        # Check if step is enabled
        if step_config and not step_config.enabled:
            click.echo(f"[{i + 1}/{len(PIPELINE_STEPS)}] {step.name} — SKIPPED (disabled in config)")
            skipped += 1
            continue

        # Build args: config args first, then CLI overrides
        if step.name in cli_override_map:
            args = cli_override_map[step.name]
        elif step_config:
            args = step_config.to_cli_args()
        else:
            args = []

        click.echo(f"[{i + 1}/{len(PIPELINE_STEPS)}] {step.name}")

        returncode = _run_script(step, args)

        if returncode != 0:
            failed += 1
            click.echo(f"Step '{step.name}' FAILED (exit code {returncode}).")
            if effective_stop:
                click.echo("Pipeline stopped due to error.")
                sys.exit(returncode)
        else:
            passed += 1
            click.echo(f"Step '{step.name}' OK.")

    log_header(title=f"Pipeline finished. Passed: {passed}, Failed: {failed}, Skipped: {skipped}")


if __name__ == "__main__":
    cli()
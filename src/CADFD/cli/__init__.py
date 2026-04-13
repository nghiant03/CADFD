"""CADFD CLI - Centralized command-line interface using Typer."""

import typer

from CADFD.cli.evaluate import app as evaluate_app
from CADFD.cli.inject import app as inject_app
from CADFD.cli.optimize import app as optimize_app
from CADFD.cli.prepare import app as prepare_app
from CADFD.cli.train import app as train_app
from CADFD.logging import configure_logging

app = typer.Typer(
    name="cadfd",
    help="CADFD - Communication-Aware Distributed Fault Diagnosis",
    no_args_is_help=True,
)


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Configure global options."""
    level = "DEBUG" if debug else "INFO"
    configure_logging(level=level, verbose=verbose)


app.add_typer(inject_app, name="inject", help="Inject faults into sensor datasets")
app.add_typer(prepare_app, name="prepare", help="Prepare dataset variants (e.g. graph topology)")
app.add_typer(train_app, name="train", help="Train deep learning models")
app.add_typer(evaluate_app, name="evaluate", help="Evaluate trained models")
app.add_typer(optimize_app, name="optimize", help="Hyperparameter optimization with Optuna")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

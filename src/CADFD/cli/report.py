"""Report aggregation CLI: compare runs across models and fault ratios."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="report", help="Aggregate and compare run artifacts")

_FAULT_RE = re.compile(r"fault(\d+)")


def _extract_fault_ratio(manifest: dict) -> str:
    dataset_path = (manifest.get("dataset") or {}).get("path", "")
    match = _FAULT_RE.search(dataset_path)
    if match:
        return f"fault{match.group(1)}"
    return "unknown"


def _iter_runs(runs_dir: Path):
    for model_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            eval_path = run_dir / "eval_metrics.json"
            manifest_path = run_dir / "manifest.json"
            if not eval_path.exists() or not manifest_path.exists():
                continue
            try:
                eval_data = json.loads(eval_path.read_text())
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                continue
            yield model_dir.name, run_dir, eval_data, manifest


@app.command("compare")
def compare(
    runs_dir: Path = typer.Argument(Path("runs"), help="Root runs directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write CSV to this path"),
    metric: str = typer.Option("macro_f1", "--metric", help="Primary metric for sorting"),
) -> None:
    """Compare metrics across (model, fault_ratio) pairs."""
    console = Console()
    rows: list[dict] = []
    for model, run_dir, ev, mf in _iter_runs(runs_dir):
        ratio = _extract_fault_ratio(mf)
        per_class = ev.get("per_class", {})
        rows.append({
            "model": model,
            "fault": ratio,
            "run_id": run_dir.name,
            "accuracy": ev.get("accuracy", float("nan")),
            "macro_f1": ev.get("macro_f1", float("nan")),
            "f1_NORMAL": per_class.get("NORMAL", {}).get("f1", float("nan")),
            "f1_SPIKE": per_class.get("SPIKE", {}).get("f1", float("nan")),
            "f1_DRIFT": per_class.get("DRIFT", {}).get("f1", float("nan")),
            "f1_STUCK": per_class.get("STUCK", {}).get("f1", float("nan")),
        })

    if not rows:
        console.print("[red]No completed runs found[/red]")
        raise typer.Exit(1)

    # Keep latest run per (model, fault); rows already sorted by run_id timestamp.
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        latest[(row["model"], row["fault"])] = row
    rows = sorted(latest.values(), key=lambda r: (r["fault"], -r.get(metric, 0.0)))

    table = Table(title=f"Run comparison ({len(rows)} runs)", show_lines=False)
    for col in ["fault", "model", "accuracy", "macro_f1", "f1_NORMAL", "f1_SPIKE", "f1_DRIFT", "f1_STUCK"]:
        table.add_column(col)
    for row in rows:
        table.add_row(
            row["fault"],
            row["model"],
            f"{row['accuracy']:.4f}",
            f"{row['macro_f1']:.4f}",
            f"{row['f1_NORMAL']:.4f}",
            f"{row['f1_SPIKE']:.4f}",
            f"{row['f1_DRIFT']:.4f}",
            f"{row['f1_STUCK']:.4f}",
        )
    console.print(table)

    if output is not None:
        cols = ["fault", "model", "run_id", "accuracy", "macro_f1",
                "f1_NORMAL", "f1_SPIKE", "f1_DRIFT", "f1_STUCK"]
        lines = [",".join(cols)]
        for row in rows:
            lines.append(",".join(str(row[c]) for c in cols))
        output.write_text("\n".join(lines) + "\n")
        console.print(f"[green]Wrote CSV to {output}[/green]")


@app.command("list")
def list_runs(runs_dir: Path = typer.Argument(Path("runs"))) -> None:
    """List all runs with their fault ratios."""
    console = Console()
    table = Table(title="Runs")
    for col in ["model", "fault", "run_id", "macro_f1"]:
        table.add_column(col)
    for model, run_dir, ev, mf in _iter_runs(runs_dir):
        table.add_row(model, _extract_fault_ratio(mf), run_dir.name, f"{ev.get('macro_f1', 0):.4f}")
    console.print(table)

"""Improvement curve plot for autoresearch runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cv_autoresearch.search.history import HistoryEntry, SearchHistory
from cv_autoresearch.types import SearchPhase, TrialStatus


def plot_improvement_curve(
    history: SearchHistory,
    primary_metric: str,
    higher_is_better: bool,
    output_path: str,
    hp_trial_count: int,
) -> None:
    """Generate and save the improvement curve plot.

    Draws a running-best metric line across all trials, with coloured
    markers and annotations only at improvement points.  A vertical
    dashed line separates the hyperparameter and augmentation phases.

    Args:
        history: Completed SearchHistory containing all trial entries.
        primary_metric: Name of the metric (used as Y-axis label).
        higher_is_better: Optimization direction.
        output_path: File path to write the PNG (parent dirs created).
        hp_trial_count: Number of hyperparameter trials, used to draw
            the phase-boundary vertical line.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe in all envs
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not installed; skip silently

    if not history.entries:
        return

    trial_ids, running_best, improvement_ids, improvement_vals, labels = (
        _build_plot_data(history.entries, higher_is_better)
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    # Running-best line
    ax.plot(trial_ids, running_best, color="#AAAAAA", linewidth=1.5,
            zorder=1, label="Running best")

    # Improvement markers
    if improvement_ids:
        ax.scatter(improvement_ids, improvement_vals, color="#2196F3",
                   s=80, zorder=3, label="Improvement")

        # Alternate annotations above/below to reduce overlap
        for i, (x, y, label) in enumerate(zip(improvement_ids, improvement_vals, labels)):
            offset = 12 if i % 2 == 0 else -18
            va = "bottom" if i % 2 == 0 else "top"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7,
                color="#1565C0",
                arrowprops=dict(arrowstyle="-", color="#90CAF9", lw=0.8),
            )

    # Phase boundary
    if hp_trial_count > 0 and hp_trial_count < len(trial_ids):
        ax.axvline(x=hp_trial_count - 0.5, color="#E53935", linewidth=1,
                   linestyle="--", alpha=0.7, label="HP → Augmentation phase")

    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_xlabel("Trial", fontsize=10)
    ax.set_ylabel(primary_metric, fontsize=10)
    ax.set_title(f"Autoresearch: {primary_metric} improvements ({direction})", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_plot_data(
    entries: list[HistoryEntry],
    higher_is_better: bool,
) -> tuple[list[int], list[float], list[int], list[float], list[str]]:
    """Compute running-best series and collect improvement points.

    Args:
        entries: All history entries in chronological order.
        higher_is_better: Optimization direction.

    Returns:
        Tuple of:
        - trial_ids: X values for all trials.
        - running_best: Running-best metric at each trial.
        - improvement_ids: X values of improvement trials only.
        - improvement_vals: Y values of improvement trials only.
        - labels: Annotation label for each improvement.
    """
    worst = float("-inf") if higher_is_better else float("inf")
    current_best = worst

    trial_ids: list[int] = []
    running_best: list[float] = []
    improvement_ids: list[int] = []
    improvement_vals: list[float] = []
    labels: list[str] = []

    for entry in entries:
        trial_ids.append(int(entry.trial_id))

        if entry.status == TrialStatus.FAILED or entry.metric_after is None:
            running_best.append(current_best if current_best != worst else float("nan"))
            continue

        if entry.improved:
            current_best = entry.metric_after
            improvement_ids.append(int(entry.trial_id))
            improvement_vals.append(entry.metric_after)
            labels.append(_make_label(entry))

        running_best.append(current_best if current_best != worst else entry.metric_after)

    return trial_ids, running_best, improvement_ids, improvement_vals, labels


def _make_label(entry: HistoryEntry) -> str:
    """Build the annotation label for an improvement point.

    Args:
        entry: An improvement HistoryEntry.

    Returns:
        Human-readable label string, e.g. ``"lr=0.001"`` or ``"baseline"``.
    """
    if entry.trial_id == 0 or entry.param_name is None:
        return "baseline"

    name = entry.param_name
    value = entry.param_value

    if entry.phase == SearchPhase.AUGMENTATION:
        # Augmentation: show transform name + enabled/disabled
        if value is None:
            return f"{name} disabled"
        return f"{name} enabled"

    # Hyperparameter: show param=value
    if isinstance(value, float):
        return f"{name}={value:.4g}"
    return f"{name}={value}"

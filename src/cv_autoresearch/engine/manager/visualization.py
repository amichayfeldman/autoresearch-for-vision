"""Final F1-over-epochs visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import struct
import zlib


def plot_f1_progress(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save F1 curve with successful baseline promotion annotations."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _write_placeholder_png(target)
        return target

    xs: list[int] = []
    ys: list[float] = []
    labels: list[tuple[int, float, str]] = []
    offset = 0
    for record in records:
        epoch_metrics = record.get("epoch_metrics") or []
        for metric in epoch_metrics:
            if "f1" in metric:
                xs.append(offset + int(metric.get("epoch", len(xs) + 1)))
                ys.append(float(metric["f1"]))
        if epoch_metrics:
            offset = xs[-1] if xs else offset
        if record.get("promoted") and record.get("primary_metric_after") is not None:
            labels.append((max(offset, 1), float(record["metrics"].get("f1", record["primary_metric_after"])), record.get("one_change_summary", "")))

    if not xs:
        xs, ys = [0], [0.0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=1.8, label="F1")
    for x, y, label in labels:
        ax.scatter([x], [y], marker="s", color="tab:green", zorder=3)
        if label:
            ax.annotate(
                label[:80],
                xy=(x, y),
                xytext=(x + 0.2, min(1.0, y + 0.08)),
                arrowprops={"arrowstyle": "->", "color": "tab:green"},
                fontsize=8,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def _write_placeholder_png(target: Path) -> None:
    """Write a tiny valid PNG when plotting dependencies are unavailable."""
    width, height = 2, 2
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    target.write_bytes(png)

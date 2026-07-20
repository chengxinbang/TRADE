"""Plot mean average return against tau from extract_tau_scores.py output."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_tau_scores(txt_path):
    """Read the aligned data rows written by extract_tau_scores.py."""
    grouped_scores = defaultdict(list)
    with open(txt_path, "r", encoding="utf-8") as txt_file:
        for line_number, line in enumerate(txt_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("tau"):
                continue
            if set(stripped) == {"-"}:
                continue

            fields = stripped.split(maxsplit=3)
            if len(fields) < 3:
                raise ValueError(f"Invalid data row at line {line_number}: {line.rstrip()}")
            try:
                tau = float(fields[0])
                score = float(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid tau or average_return at line {line_number}: {line.rstrip()}"
                ) from exc
            if not np.isfinite(tau) or not np.isfinite(score):
                continue
            grouped_scores[tau].append(score)

    if not grouped_scores:
        raise ValueError("No valid tau and average_return data were found in the txt file.")
    return grouped_scores


def plot_tau_scores(txt_path, output_path=None, font_size=12.0):
    """Create a tau-versus-average-return line plot with standard-deviation bars."""
    txt_path = Path(txt_path).expanduser().resolve()
    if not txt_path.is_file():
        raise FileNotFoundError(f"Txt file does not exist: {txt_path}")
    if font_size <= 0:
        raise ValueError("font_size must be greater than zero.")

    grouped_scores = _read_tau_scores(txt_path)
    # Display tau values in ascending numerical order.
    ordered_taus = sorted(grouped_scores)
    taus = np.asarray(ordered_taus, dtype=np.float64)
    means = np.asarray([np.mean(grouped_scores[tau]) for tau in taus], dtype=np.float64)
    errors = np.asarray(
        [np.std(grouped_scores[tau], ddof=1) if len(grouped_scores[tau]) > 1 else 0.0 for tau in taus],
        dtype=np.float64,
    )

    if output_path is None:
        output_path = txt_path.with_name("tau_average_return_curve.pdf")
    else:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tau includes zero, so a logarithmic axis is not valid. Use evenly spaced
    # positions and show the exact tau values as tick labels to keep them legible.
    x_positions = np.arange(len(taus), dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_positions,
        means,
        "-o",
        color="#EAB883",
        linewidth=2.0,
        markersize=4,
        zorder=2,
    )
    plt.errorbar(
        x_positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#4f6d7a",
        elinewidth=1.8,
        capsize=5,
        zorder=3,
    )
    plt.xticks(x_positions, [f"{tau:.12g}" for tau in taus])
    plt.xlabel("τ")
    plt.ylabel("Average Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path, len(taus)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average return versus tau with standard-deviation error bars."
    )
    parser.add_argument("txt_file", help="Txt file created by extract_tau_scores.py.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: tau_average_return_curve.png next to the txt file.",
    )
    parser.add_argument(
        "--font-size",
        "--font_size",
        dest="font_size",
        type=float,
        default=20.0,
        help="Font size for all plot text. Default: 12.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path, tau_count = plot_tau_scores(args.txt_file, args.output, args.font_size)
    print(f"Plotted {tau_count} tau value(s).")
    print(f"Wrote figure: {output_path}")


if __name__ == "__main__":
    main()

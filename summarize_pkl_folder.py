import argparse
import math
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


SEED_PATTERN = re.compile(r"seed(\d+)")


def _load_pickle_compat(pkl_path):
    with open(pkl_path, "rb") as f_pkl:
        try:
            return pickle.load(f_pkl)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") and exc.name.startswith("numpy._core"):
                sys.modules.setdefault("numpy._core", np.core)
                sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
                f_pkl.seek(0)
                return pickle.load(f_pkl)
            raise


def _infer_seed(data, pkl_path):
    if isinstance(data, dict) and "seed" in data:
        try:
            return int(data["seed"])
        except (TypeError, ValueError):
            pass

    match = SEED_PATTERN.search(os.path.basename(pkl_path))
    if match:
        return int(match.group(1))
    return None


def _mean_return(data):
    if not isinstance(data, dict):
        return None, 0

    returns = np.asarray(data.get("returns", []), dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return None, 0
    return float(np.mean(returns)), int(returns.size)


def _elapsed_seconds(data):
    if not isinstance(data, dict):
        return None

    if "elapsed_time_sec" in data:
        try:
            return float(data["elapsed_time_sec"])
        except (TypeError, ValueError):
            return None

    if "elapsed_time_hours" in data:
        try:
            return float(data["elapsed_time_hours"]) * 3600.0
        except (TypeError, ValueError):
            return None

    return None


def _fmt_number(value, digits=6):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _summarize_folder(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    pkl_paths = sorted(folder.glob("*.pkl"))
    rows = []
    for pkl_path in pkl_paths:
        data = _load_pickle_compat(pkl_path)
        mean_score, num_returns = _mean_return(data)
        elapsed_sec = _elapsed_seconds(data)
        rows.append(
            {
                "file": pkl_path.name,
                "seed": _infer_seed(data, str(pkl_path)),
                "mean_score": mean_score,
                "num_returns": num_returns,
                "elapsed_sec": elapsed_sec,
            }
        )
    return folder, rows


def _write_txt(folder, rows, output_name):
    output_path = folder / output_name
    valid_scores = [row["mean_score"] for row in rows if row["mean_score"] is not None]
    valid_elapsed = [row["elapsed_sec"] for row in rows if row["elapsed_sec"] is not None]

    avg_score = float(np.mean(valid_scores)) if valid_scores else None
    avg_elapsed = float(np.mean(valid_elapsed)) if valid_elapsed else None

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Folder: {folder}\n")
        f.write(f"PKL files: {len(rows)}\n\n")
        f.write("Per-file seed summary\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'file':50s} {'seed':>8s} {'mean_score':>16s} "
            f"{'num_returns':>12s} {'elapsed_sec':>16s} {'elapsed_hours':>16s}\n"
        )
        f.write("-" * 100 + "\n")

        for row in rows:
            elapsed_hours = None
            if row["elapsed_sec"] is not None:
                elapsed_hours = row["elapsed_sec"] / 3600.0
            seed = "N/A" if row["seed"] is None else str(row["seed"])
            f.write(
                f"{row['file'][:50]:50s} {seed:>8s} "
                f"{_fmt_number(row['mean_score']):>16s} "
                f"{row['num_returns']:12d} "
                f"{_fmt_number(row['elapsed_sec'], digits=3):>16s} "
                f"{_fmt_number(elapsed_hours, digits=6):>16s}\n"
            )

        f.write("-" * 100 + "\n")
        f.write("Overall seed averages\n")
        f.write(f"average_mean_score: {_fmt_number(avg_score)}\n")
        f.write(f"average_elapsed_sec: {_fmt_number(avg_elapsed, digits=3)}\n")
        avg_elapsed_hours = None if avg_elapsed is None else avg_elapsed / 3600.0
        f.write(f"average_elapsed_hours: {_fmt_number(avg_elapsed_hours, digits=6)}\n")

        if not valid_scores and rows:
            f.write("\nNote: No valid returns were found, so average_mean_score is N/A.\n")

    return output_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize all pkl files in one folder into a txt report."
    )
    parser.add_argument(
        "folder",
        help="Folder containing pkl files.",
    )
    parser.add_argument(
        "--output-name",
        default="pkl_seed_summary.txt",
        help="Txt file name written inside the folder.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    folder, rows = _summarize_folder(args.folder)
    output_path = _write_txt(folder, rows, args.output_name)
    print(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()

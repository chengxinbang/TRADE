"""Write the seed, mean return, and elapsed training time for pkl files."""

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
    """Load pickles created with either NumPy 1.x or 2.x module paths."""
    with open(pkl_path, "rb") as pkl_file:
        try:
            return pickle.load(pkl_file)
        except ModuleNotFoundError as exc:
            if not (getattr(exc, "name", "") or "").startswith("numpy._core"):
                raise
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            pkl_file.seek(0)
            return pickle.load(pkl_file)


def _get_seed(data, pkl_path):
    if isinstance(data, dict):
        try:
            return int(data["seed"])
        except (KeyError, TypeError, ValueError):
            pass

    match = SEED_PATTERN.search(os.path.basename(pkl_path))
    return int(match.group(1)) if match else None


def _get_mean_score(data):
    if not isinstance(data, dict):
        return None

    scores = np.asarray(data.get("returns", []), dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    return float(np.mean(scores)) if scores.size else None


def _get_elapsed_hours(data):
    if not isinstance(data, dict):
        return None

    try:
        if "elapsed_time_hours" in data:
            elapsed_hours = float(data["elapsed_time_hours"])
        elif "elapsed_time_sec" in data:
            elapsed_hours = float(data["elapsed_time_sec"]) / 3600.0
        else:
            return None
    except (TypeError, ValueError):
        return None

    return elapsed_hours if math.isfinite(elapsed_hours) else None


def _format(value, digits=6):
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_time_report(folder, output_name="time_summary.txt"):
    """Create a txt report for all pkl files directly inside ``folder``."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    rows = []
    for pkl_path in sorted(folder.glob("*.pkl")):
        data = _load_pickle_compat(pkl_path)
        rows.append(
            (
                pkl_path.name,
                _get_seed(data, pkl_path),
                _get_mean_score(data),
                _get_elapsed_hours(data),
            )
        )

    output_path = folder / output_name
    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Folder: {folder}\n")
        txt_file.write(f"PKL files: {len(rows)}\n\n")
        txt_file.write(f"{'pkl_file':50s} {'seed':>8s} {'mean_score':>16s} {'elapsed_hours':>16s}\n")
        txt_file.write("-" * 96 + "\n")
        for filename, seed, mean_score, elapsed_hours in rows:
            seed_text = "N/A" if seed is None else str(seed)
            txt_file.write(
                f"{filename[:50]:50s} {seed_text:>8s} "
                f"{_format(mean_score):>16s} {_format(elapsed_hours):>16s}\n"
            )

    return output_path, len(rows)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract seed, mean return, and elapsed hours from pkl files."
    )
    parser.add_argument("folder", help="Folder containing the pkl files.")
    parser.add_argument(
        "--output-name",
        default="time_summary.txt",
        help="Txt filename to create inside the target folder.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path, pkl_count = write_time_report(args.folder, args.output_name)
    print(f"Processed {pkl_count} pkl file(s).")
    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()

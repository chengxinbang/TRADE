"""Extract per-pkl tau, seed, and mean return values into an aligned txt file."""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np


SEED_PATTERN = re.compile(r"seed(\d+)")
TAU_DIR_PATTERN = re.compile(r"^tau_(.+)$")


def _load_pickle_compat(pkl_path):
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


def _decode_tau_label(label):
    """Decode sensitivity folder names such as tau_0p0001 and tau_1em05."""
    value = label.replace("p", ".")
    value = re.sub(r"e[mM]", "e-", value)
    if value.startswith("m"):
        value = "-" + value[1:]
    return float(value)


def _get_tau(data, pkl_path, root_folder):
    if isinstance(data, dict):
        try:
            return float(data["adaptive_epsilon_tol"])
        except (KeyError, TypeError, ValueError):
            pass

    relative_parts = pkl_path.relative_to(root_folder).parts[:-1]
    for part in reversed(relative_parts):
        match = TAU_DIR_PATTERN.fullmatch(part)
        if match:
            return _decode_tau_label(match.group(1))
    return None


def _get_seed(data, pkl_path):
    if isinstance(data, dict):
        try:
            return int(data["seed"])
        except (KeyError, TypeError, ValueError):
            pass

    match = SEED_PATTERN.search(pkl_path.name)
    return int(match.group(1)) if match else None


def _get_mean_return(data):
    if not isinstance(data, dict):
        return None
    returns = np.asarray(data.get("returns", []), dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    return float(np.mean(returns)) if returns.size else None


def _format_tau(value):
    return "N/A" if value is None else f"{value:.12g}"


def _format_score(value):
    return "N/A" if value is None else f"{value:.6f}"


def extract_tau_scores(folder, output_name="tau_seed_scores.txt"):
    """Recursively extract all pkl records below ``folder`` into a txt file."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    rows = []
    for pkl_path in sorted(folder.rglob("*.pkl")):
        data = _load_pickle_compat(pkl_path)
        tau = _get_tau(data, pkl_path, folder)
        seed = _get_seed(data, pkl_path)
        mean_return = _get_mean_return(data)
        if tau is None:
            print(f"[Skip] Tau not found: {pkl_path}")
            continue
        if seed is None:
            print(f"[Skip] Seed not found: {pkl_path}")
            continue
        if mean_return is None:
            print(f"[Skip] No valid returns: {pkl_path}")
            continue
        rows.append((tau, seed, mean_return, str(pkl_path.relative_to(folder))))

    rows.sort(key=lambda row: (row[0], row[1], row[3]))
    output_path = folder / output_name
    tau_width = max(12, len("tau"), *(len(_format_tau(row[0])) for row in rows))
    seed_width = max(8, len("seed"), *(len(str(row[1])) for row in rows))
    score_width = max(18, len("average_return"))
    file_width = max(20, len("pkl_file"), *(len(row[3]) for row in rows))
    line_width = tau_width + seed_width + score_width + file_width + 9

    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"# source_folder: {folder}\n")
        txt_file.write(f"# valid_pkl_records: {len(rows)}\n\n")
        txt_file.write(
            f"{'tau':>{tau_width}s}  {'seed':>{seed_width}s}  "
            f"{'average_return':>{score_width}s}  {'pkl_file':<{file_width}s}\n"
        )
        txt_file.write("-" * line_width + "\n")
        for tau, seed, mean_return, relative_path in rows:
            txt_file.write(
                f"{_format_tau(tau):>{tau_width}s}  {seed:>{seed_width}d}  "
                f"{_format_score(mean_return):>{score_width}s}  {relative_path:<{file_width}s}\n"
            )

    return output_path, len(rows)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tau, seed, and mean return from all pkl files below a folder."
    )
    parser.add_argument("folder", help="Root folder containing tau experiment result folders.")
    parser.add_argument(
        "--output-name",
        default="tau_seed_scores.txt",
        help="Txt filename to create inside the root folder.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path, count = extract_tau_scores(args.folder, args.output_name)
    print(f"Processed {count} valid pkl record(s).")
    print(f"Wrote data file: {output_path}")


if __name__ == "__main__":
    main()

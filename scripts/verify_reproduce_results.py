import argparse
import csv
import math
from pathlib import Path

import numpy as np


DATASETS = [
    10, 11, 14, 22, 25, 29, 31, 46, 51, 54, 151, 334, 470, 846, 934,
    1043, 1067, 1459, 1486, 1489, 1493, 40536, 40981, 41027, 41143,
]
SEEDS = [1, 2, 3, 4, 5]
MODELS = [
    "randomforest", "xgboost", "catboost", "lightgbm", "mlp",
    "embedmlp", "mlpplr", "resnet", "ftt", "t2gformer", "saint",
    "modernnca",
]


def finite_array(value):
    array = np.asarray(value)
    return array.size > 0 and np.issubdtype(array.dtype, np.number) and np.isfinite(array).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing seed=*/data=* result directories")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    report = args.report or root.parent / "reproduce_metrics.csv"
    missing = []
    invalid = []
    rows = []
    revisions = set()

    for seed in SEEDS:
        for dataset in DATASETS:
            for model in MODELS:
                path = root / f"seed={seed}" / f"data={dataset}" / f"model={model}..init_hps=False..deep=0..hyper=0.npy"
                if not path.is_file():
                    missing.append(path)
                    continue

                try:
                    saved = np.load(path, allow_pickle=True).item()
                    if not isinstance(saved, dict):
                        raise ValueError(f"expected dict, found {type(saved).__name__}")
                    if "Prediction" not in saved or not finite_array(saved["Prediction"]):
                        raise ValueError("Prediction is missing, empty, or non-finite")
                    performance = saved.get("Performance")
                    if not isinstance(performance, dict) or not performance:
                        raise ValueError("Performance is missing or empty")

                    row = {"dataset": dataset, "seed": seed, "model": model, "path": str(path)}
                    for name, value in performance.items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            value = float(value)
                            if not math.isfinite(value):
                                raise ValueError(f"non-finite metric: {name}")
                        row[name] = value
                    rows.append(row)

                    revision = saved.get("implementation_id")
                    if revision is not None:
                        revisions.add(str(revision))
                except Exception as error:
                    invalid.append((path, str(error)))

    fieldnames = ["dataset", "seed", "model"]
    metric_names = sorted({key for row in rows for key in row if key not in {"dataset", "seed", "model", "path"}})
    fieldnames.extend(metric_names)
    fieldnames.append("path")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    expected = len(DATASETS) * len(SEEDS) * len(MODELS)
    print(f"Root: {root}")
    print(f"Valid: {len(rows)}/{expected}")
    print(f"Missing: {len(missing)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Implementation IDs: {len(revisions)}")
    print(f"Metrics report: {report}")

    for path in missing[:20]:
        print(f"MISSING {path}")
    for path, error in invalid[:20]:
        print(f"INVALID {path}: {error}")
    if len(missing) > 20 or len(invalid) > 20:
        print("Only the first 20 missing and invalid files are shown.")

    raise SystemExit(1 if missing or invalid or len(revisions) > 1 else 0)


if __name__ == "__main__":
    main()

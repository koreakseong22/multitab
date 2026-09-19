import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import warnings

import joblib
import numpy as np
import openml
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libs.data import load_data, split_data
from libs.eval import calculate_auroc, calculate_multi_auroc


DATASETS = [
    10, 11, 14, 22, 25, 29, 31, 46, 51, 54, 151, 334, 470,
    846, 934, 1043, 1067, 1459, 1489, 1493, 40981, 41027, 41143,
]
TABR_HPO_DATASETS = [dataset for dataset in DATASETS if dataset not in {151, 41027}]
MODELS = [
    "randomforest", "xgboost", "catboost", "lightgbm", "mlp",
    "embedmlp", "mlpplr", "resnet", "ftt", "t2gformer", "saint",
    "modernnca",
]
RESULT_SUFFIX = "..init_hps=False..deep=0..hyper=0.npy"
OFFICIAL_MULTITAB_COMMIT = "b40c74d9be3e315b7e0ad5dfe475af57bddb7bab"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reproduce-archive",
        type=Path,
        default=Path("final_reproduce_23datasets_no_tabr.tar.gz"),
    )
    parser.add_argument(
        "--tabr-hpo-archive",
        type=Path,
        default=Path("tabr_hpo_2080ti_21datasets_seed1-5.tar.gz"),
    )
    parser.add_argument("--openml-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def official_auroc(y_true, scores, tasktype):
    scores = np.asarray(scores)
    if tasktype == "binclass":
        probabilities = expit(scores)
        return calculate_auroc(np.asarray(y_true).reshape(-1), probabilities)
    if tasktype == "multiclass":
        probabilities = softmax(scores, axis=1)
        return calculate_multi_auroc(np.asarray(y_true), probabilities)
    raise ValueError(f"AUROC is not defined for task type {tasktype!r}")


def load_target_results(archive):
    expected = {
        f"reproduce_logs/seed={seed}/data={dataset}/model={model}{RESULT_SUFFIX}"
        for seed in range(1, 6)
        for dataset in DATASETS
        for model in MODELS
    }
    results = {}
    duplicates = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or member.name not in expected:
                continue
            if member.name in results:
                duplicates.append(member.name)
                continue
            payload = handle.extractfile(member).read()
            results[member.name] = np.load(
                io.BytesIO(payload), allow_pickle=True
            ).item()
    missing = sorted(expected - results.keys())
    if missing or duplicates:
        raise RuntimeError(
            f"reproduce archive mismatch: missing={len(missing)}, "
            f"duplicates={len(duplicates)}"
        )
    return results


def recalculate_reproduce(results, data_info, output_path):
    rows = []
    for dataset in DATASETS:
        tasktype = data_info[str(dataset)]["tasktype"]
        X, y, _, _, _ = load_data(dataset)
        for seed in range(1, 6):
            (_, _), (_, _), (_, y_test), _ = split_data(
                X, y, tasktype, seed=seed, device="cpu"
            )
            y_test = y_test.numpy()
            for model in MODELS:
                name = (
                    f"reproduce_logs/seed={seed}/data={dataset}/"
                    f"model={model}{RESULT_SUFFIX}"
                )
                result = results[name]
                prediction = np.asarray(result["Prediction"])
                scores = np.asarray(result["Probability"])
                if prediction.shape[0] != y_test.shape[0]:
                    raise ValueError(
                        f"test length mismatch for {name}: "
                        f"prediction={prediction.shape[0]}, y={y_test.shape[0]}"
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    recalculated = finite_float(
                        official_auroc(y_test, scores, tasktype)
                    )
                saved = finite_float(
                    result.get("Performance", {}).get("auroc_test")
                )
                if recalculated is None and saved is None:
                    difference = None
                    status = "both_undefined"
                elif recalculated is None or saved is None:
                    difference = None
                    status = "definedness_mismatch"
                else:
                    difference = recalculated - saved
                    status = "match" if abs(difference) <= 1e-12 else "mismatch"
                rows.append({
                    "data_id": dataset,
                    "seed": seed,
                    "model": model,
                    "tasktype": tasktype,
                    "n_test": y_test.shape[0],
                    "auroc_official_recalculated": recalculated,
                    "auroc_saved": saved,
                    "difference": difference,
                    "status": status,
                    "implementation_id": result.get("implementation_id"),
                })

    write_csv(output_path, rows)
    return rows


def audit_tabr_hpo(archive, output_path):
    expected = {
        f"optim_logs/seed={seed}/data={dataset}..model=tabr.pkl"
        for seed in range(1, 6)
        for dataset in TABR_HPO_DATASETS
    }
    rows = []
    seen = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or member.name not in expected:
                continue
            seen.add(member.name)
            study = joblib.load(io.BytesIO(handle.extractfile(member).read()))
            complete = [
                trial for trial in study.trials if trial.state.name == "COMPLETE"
            ]
            best = study.best_trial if complete else None
            optimal = (
                best is not None
                and getattr(study.direction, "name", "") == "MAXIMIZE"
                and study.best_value >= 1.0
            )
            seed = int(member.name.split("seed=")[1].split("/")[0])
            dataset = int(member.name.split("data=")[1].split("..")[0])
            rows.append({
                "data_id": dataset,
                "seed": seed,
                "complete_trials": len(complete),
                "ready": len(complete) >= 100 or optimal,
                "best_trial_number": None if best is None else best.number,
                "best_validation_objective": None if best is None else best.value,
                "stored_best_trial_auroc_test": (
                    None if best is None else best.user_attrs.get("auroc_test")
                ),
                "auroc_recalculated": False,
                "reason": "HPO archive has no raw test logits or predictions",
                "implementation_id": study.user_attrs.get("implementation_id"),
            })
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"TabR HPO archive is missing {len(missing)} studies")
    rows.sort(key=lambda row: (row["seed"], row["data_id"]))
    write_csv(output_path, rows)
    return rows


def main():
    args = parse_args()
    if args.openml_cache is not None:
        openml.config.set_root_cache_directory(args.openml_cache)
    with Path("dataset_id.json").open(encoding="utf-8") as handle:
        data_info = json.load(handle)

    results = load_target_results(args.reproduce_archive)
    reproduce_output = args.output_dir / "official_auroc_23datasets_no_tabr.csv"
    reproduce_rows = recalculate_reproduce(
        results, data_info, reproduce_output
    )
    hpo_output = args.output_dir / "tabr_hpo_21datasets_audit.csv"
    hpo_rows = audit_tabr_hpo(args.tabr_hpo_archive, hpo_output)

    statuses = {}
    for row in reproduce_rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    mismatch_output = args.output_dir / "official_auroc_mismatches.csv"
    mismatches = [
        row for row in reproduce_rows
        if row["status"] in {"mismatch", "definedness_mismatch"}
    ]
    write_csv(mismatch_output, mismatches)
    summary_output = args.output_dir / "official_auroc_recalculation_summary.json"
    summary = {
        "official_multitab_commit": OFFICIAL_MULTITAB_COMMIT,
        "method": {
            "binclass": "sigmoid(logits), positive-class ROC AUC",
            "multiclass": "softmax(logits), macro one-vs-rest ROC AUC",
            "undefined": "None when a test split lacks a required class",
        },
        "reproduce_archive": {
            "path": str(args.reproduce_archive),
            "sha256": sha256(args.reproduce_archive),
            "expected_rows": len(DATASETS) * 5 * len(MODELS),
            "recalculated_rows": len(reproduce_rows),
            "comparison_to_saved": statuses,
        },
        "tabr_hpo_archive": {
            "path": str(args.tabr_hpo_archive),
            "sha256": sha256(args.tabr_hpo_archive),
            "expected_studies": len(TABR_HPO_DATASETS) * 5,
            "audited_studies": len(hpo_rows),
            "ready_studies": sum(bool(row["ready"]) for row in hpo_rows),
            "recalculated_auroc": 0,
            "limitation": "HPO studies do not contain raw test logits or predictions",
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Reproduce rows: {len(reproduce_rows)}")
    print(f"AUROC comparison: {statuses}")
    print(f"Reproduce CSV: {reproduce_output}")
    print(f"Mismatch CSV: {mismatch_output}")
    print(f"TabR HPO studies: {len(hpo_rows)}")
    print(f"TabR HPO ready: {sum(bool(row['ready']) for row in hpo_rows)}")
    print("TabR AUROC recalculated: 0 (raw logits are not stored in HPO logs)")
    print(f"TabR HPO audit CSV: {hpo_output}")
    print(f"Summary JSON: {summary_output}")


if __name__ == "__main__":
    main()

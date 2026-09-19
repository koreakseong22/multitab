"""Inference latency of tuned TabR, timed with TabERA's protocol module.

Run from the multitab repository root with its own environment:

    python scripts/bench_inference_tabr.py --openml_id 1489 --seed 1 --gpu_id 0 \
        --tabera_dir ../TabERA --out ../TabERA/results/inference_latency

The protocol code (libs/inference_timing.py) is loaded from --tabera_dir by
path, so TabR and TabERA are timed by the same function. The model is trained
once exactly as reproduce.py --mode best does (best trial of the HPO study,
rearrange_params, same split), because the archives keep no checkpoints.

What is timed: one TabR forward on a batch with the candidate keys already
encoded and the search index already built -- the state multitab's
predict() reaches before its chunk loop. Candidate encoding and index
construction are excluded, as they are for TabERA's memory bank. Note that
multitab's public predict()/predict_proba() rebuild both on every call; that
API-level cost is reported separately as api_predict_proba.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_protocol(tabera_dir):
    path = Path(tabera_dir).resolve() / "libs" / "inference_timing.py"
    spec = importlib.util.spec_from_file_location("inference_timing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openml_id", type=int, required=True)
    p.add_argument("--seed", type=int, choices=range(10), default=1)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--savepath", default=".", help="root holding optim_logs/seed=*/data=*..model=tabr.pkl")
    p.add_argument("--study_path", default=None,
                   help="explicit TabR study .pkl to take best_params from, for datasets without their own "
                        "HPO study (the four largest MultiTab sets). Recorded in the JSON as params_borrowed_from.")
    p.add_argument("--tabera_dir", default=str(ROOT.parent / "TabERA"))
    p.add_argument("--out", default=None, help="default: <tabera_dir>/results/inference_latency")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 512])
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--full_pass_repeats", type=int, default=20)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def run(args):
    import joblib
    import torch
    from libs.runtime import configure_device
    from libs.data import TabularDataset
    from libs.model import getmodel
    from libs.search_space import rearrange_params
    import libs.tabr as tabr_lib

    protocol, protocol_path = load_protocol(args.tabera_dir)
    if args.threads:
        torch.set_num_threads(args.threads)
    with open(ROOT / "dataset_id.json", encoding="utf-8") as f:
        info = json.load(f)[str(args.openml_id)]
    task = info["tasktype"]
    out_dir = Path(args.out) if args.out else Path(args.tabera_dir) / "results" / "inference_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model=tabr..data={args.openml_id}..seed={args.seed}.json"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path}")
        return

    own_path = Path(args.savepath) / f"optim_logs/seed={args.seed}/data={args.openml_id}..model=tabr.pkl"
    study_path = Path(args.study_path) if args.study_path else own_path
    if not study_path.is_file():
        raise FileNotFoundError(f"Missing {study_path}")
    study = joblib.load(study_path)
    borrowed = None
    if args.study_path and study_path.resolve() != own_path.resolve():
        borrowed = study_path.name
    device_str = configure_device(args.gpu_id)
    device = torch.device(device_str)
    dataset = TabularDataset(args.openml_id, task, device=device_str, seed=args.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    params = rearrange_params("tabr", args.openml_id, study.best_params, num_features=X_train.shape[1])
    output_dim = y_train.shape[1] if task == "multiclass" else 1
    method = getmodel("tabr", params, task, dataset, args.openml_id, X_train.shape[1], output_dim, device_str)

    t0 = time.perf_counter()
    method.fit(X_train, y_train, X_val, y_val)
    fit_s = time.perf_counter() - t0
    model = method.model.eval()

    # Set up the inference state once, exactly as predict() does before its loop.
    with torch.no_grad():
        cand_num = method.N[:50000].float() if method.N is not None else None
        cand_cat = method.C[:50000].float() if method.C is not None else None
        cand_y = method.y[:50000]
        if method.is_regression:
            cand_y = cand_y.float()
        model.cached_candidate_k = model._encode(cand_num, cand_cat)[1]
        model.cached_candidate_y = cand_y
        model.search_index = None
        model.update_index()
    n_num = method.n_num_features
    num_cols, cat_cols = method.num_cols, method.cat_features
    faiss = getattr(tabr_lib, "faiss", None)
    index_backend = type(model.search_index).__name__
    faiss_gpu = bool(faiss is not None and hasattr(faiss, "StandardGpuResources")
                     and model.cached_candidate_k.is_cuda)

    def forward(xb):
        N = xb[:, num_cols].float() if len(num_cols) else None
        C = xb[:, cat_cols].float() if len(cat_cols) else None
        x = N if C is None else (C if N is None else torch.cat([N, C], dim=1))
        return model(x_num=x[:, :n_num], x_cat=x[:, n_num:], y=None,
                     candidate_x_num=cand_num, candidate_x_cat=cand_cat, candidate_y=cand_y,
                     context_size=method.context_size, is_train=False)

    X = (X_test if args.split == "test" else X_val).to(device)
    resident = protocol.resident_memory_mb(device)
    timing = protocol.run_protocol({"prediction": forward}, X, args.batch_sizes, args.repeats,
                                   args.warmup, full_pass_batch=max(args.batch_sizes),
                                   full_pass_repeats=args.full_pass_repeats, seed=args.seed)
    api = protocol.time_full_pass(lambda xb: method.predict_proba(xb, logit=True), X,
                                  batch_size=len(X), repeats=args.full_pass_repeats)
    timing["api_predict_proba"] = {"full_pass": api,
                                   "note": "multitab predict_proba re-encodes candidates and rebuilds the index per call"}

    payload = {
        "model": "tabr",
        "dataset_id": args.openml_id, "dataset": info.get("fullname"), "tasktype": task,
        "fold": args.seed, "split": args.split,
        "n_train": int(len(y_train)), "n_eval": int(len(X)), "n_features": int(X_train.shape[1]),
        "n_candidates": int(model.cached_candidate_k.shape[0]), "context_size": method.context_size,
        "index_backend": index_backend, "faiss_gpu": faiss_gpu,
        "study": str(study_path), "trial": study.best_trial.number, "params": params,
        "params_borrowed_from": borrowed,
        "fit_s_not_protocol": fit_s,
        "protocol": {"batch_sizes": args.batch_sizes, "repeats": args.repeats, "warmup": args.warmup,
                     "full_pass_repeats": args.full_pass_repeats, "module": str(protocol_path),
                     "excluded": ["data loading", "input host->device copy", "model construction",
                                  "training", "candidate encoding", "index construction"]},
        "resident_mb_after_setup": resident,
        "environment": protocol.environment(device),
        "timing": timing,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(out_path)
    for mode, rec in timing.items():
        for key, st in rec.items():
            if isinstance(st, dict) and "ms_median" in st:
                print(f"  {mode:22s} {key:10s} rows={st['batch_rows']:5d} "
                      f"median={st['ms_median']:8.3f} ms  p90={st['ms_p90']:8.3f}  "
                      f"{st['samples_per_s']:10.0f} samples/s")


if __name__ == "__main__":
    run(parser().parse_args())

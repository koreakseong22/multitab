import pandas as pd
import numpy as np
import os, json, sys, argparse
from pathlib import Path
from libs.runtime import configure_device, implementation_id, check_implementation
from libs.data import TabularDataset
from libs.eval import calculate_metric
from scipy.special import expit, softmax
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

opts = {
    "deep": ["deep=0..hyper=0", "deep=1..hyper=0", "deep=2..hyper=0", "deep=3..hyper=0", "deep=4..hyper=0"],
    "hyper": ["deep=0..hyper=0", "deep=0..hyper=1", "deep=0..hyper=2", "deep=0..hyper=3", "deep=0..hyper=4"],
    "all": ["deep=0..hyper=0", "deep=1..hyper=0", "deep=2..hyper=0", "deep=3..hyper=0", "deep=4..hyper=0", "deep=0..hyper=1", "deep=0..hyper=2", "deep=0..hyper=3", "deep=0..hyper=4"]
}


def get_results(result_fname, dataset, tasktype, seed, data_id, model, ensemble_type="deep", basepath=".", allow_legacy_logs=False):
    result_fname = os.path.join(basepath, result_fname)
    revision = implementation_id()
    os.makedirs(os.path.dirname(result_fname) or ".", exist_ok=True)
    log_directory = os.path.join(basepath, "ensemble_logs", f"seed={seed}", f"data={data_id}")
    os.makedirs(log_directory, exist_ok=True)
    
    try:
        result = pd.read_csv(result_fname, index_col=0)
    except FileNotFoundError:
        result = pd.DataFrame(columns=("data", "task", "model", "seed", "ensemble_deep", "ensemble_hyper", "acc_rmse", "auroc_rmse", "logloss_rmse", "init"))

    if ensemble_type == "all":
        n_models = [9] 
    else:
        n_models = [2, 3, 4, 5] 
    for e in n_models:
        deep_count = 5 if ensemble_type == "all" else (e if ensemble_type == "deep" else 0)
        hyper_count = 5 if ensemble_type == "all" else (e if ensemble_type == "hyper" else 0)
        existing = ((result["data"].astype(str) == str(data_id)) &
                    (result["model"] == model) & (result["seed"] == seed) &
                    (result["ensemble_deep"] == deep_count) &
                    (result["ensemble_hyper"] == hyper_count) &
                    (result["init"].astype(str).str.lower() == "false"))
        result = result.loc[~existing].reset_index(drop=True)
        row = len(result)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
        y_std = dataset.y_std
        preds = []
        for i in range(e):
            fname = f'{basepath}/reproduce_logs/seed={seed}/data={data_id}/model={model}..init_hps=False..{opts[ensemble_type][i]}.npy'
            if not os.path.exists(fname):
                continue
            f = np.load(fname, allow_pickle=True).item()
            if not isinstance(f, str):
                check_implementation(f, revision, allow_legacy_logs)
                if tasktype == "regression":
                    preds.append(f["Prediction"])
                else:
                    preds.append(f["Probability"])

        if len(preds) != e:
            print(f"Incomplete ensemble: data={data_id}, seed={seed}, model={model}, {ensemble_type}={e}")
            continue
        preds = np.stack(preds)
        preds = np.nanmean(preds, axis=0)
        
        if tasktype == "regression":
            y_test = y_test*y_std
            preds = preds*y_std
            pred_classes = preds
        elif (tasktype == "binclass") & (model in ["modernnca"]):
            assert preds.min() >= 0
            assert preds.max() <= 1
            pred_classes = np.round(preds)
        elif (tasktype == "binclass"):
            preds = expit(preds)
            pred_classes = np.round(preds)
        else:
            preds = softmax(preds, axis=1)
            pred_classes = np.argmax(preds, axis=1)

        perf = calculate_metric(y_test, pred_classes, preds, tasktype, "test", prob=True)

        print(perf)
        
        if ensemble_type == "deep":
            if tasktype == "regression":
                result.loc[row] = [data_id, tasktype, model, seed, e, 0,
                                   perf.get("rmse_test", None), perf.get("rmse_test", None), perf.get("rmse_test", None), False]
            else:
                result.loc[row] = [data_id, tasktype, model, seed, e, 0,
                                   perf.get("acc_test", None), perf.get("auroc_test", None), perf.get("logloss_test", None), False]
        elif ensemble_type == "hyper":
            if tasktype == "regression":
                result.loc[row] = [data_id, tasktype, model, seed, 0, e,
                                   perf.get("rmse_test", None), perf.get("rmse_test", None), perf.get("rmse_test", None), False]
            else:
                result.loc[row] = [data_id, tasktype, model, seed, 0, e,
                                   perf.get("acc_test", None), perf.get("auroc_test", None), perf.get("logloss_test", None), False]
        else:
            if tasktype == "regression":
                result.loc[row] = [data_id, tasktype, model, seed, 5, 5,
                                   perf.get("rmse_test", None), perf.get("rmse_test", None), perf.get("rmse_test", None), False]
            else:
                result.loc[row] = [data_id, tasktype, model, seed, 5, 5,
                                   perf.get("acc_test", None), perf.get("auroc_test", None), perf.get("logloss_test", None), False]
        
        # print(result.loc[row].values)
        print(result.tail(1))
        
        result.to_csv(result_fname)
        if ensemble_type == "deep":
            np.save(
                f'{basepath}/ensemble_logs/seed={seed}/data={data_id}/model={model}..init_hps=False..deep={e}..hyper=0.npy',
            preds)
        elif ensemble_type == "hyper":
            np.save(
                f'{basepath}/ensemble_logs/seed={seed}/data={data_id}/model={model}..init_hps=False..deep=0..hyper={e}.npy',
            preds)
        else:
            np.save(
                f'{basepath}/ensemble_logs/seed={seed}/data={data_id}/model={model}..init_hps=False..deep=5..hyper=5.npy',
            preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--openml_id", type=int, default=None)
    parser.add_argument("--seed", type=int, choices=range(10), default=None)
    parser.add_argument("--savepath", type=str, default=".")
    parser.add_argument("--allow_legacy_logs", action="store_true")
    args = parser.parse_args()
    basepath = args.savepath
    device = configure_device(args.gpu_id)

    with open(Path(__file__).resolve().parent / 'dataset_id.json', 'r') as file:
        data_info = json.load(file)

    pd.set_option("display.float_format", "{:.3f}".format)


    result_fname = f"ensemble_results_logloss.csv"
    models = ["mlp", "embedmlp", "mlpplr", "ftt", "resnet", "t2gformer", "saint", "tabr", "modernnca"]
    datalist = "25 461 210 466 42665 444 497 10 1099 48 338 40916 23381 4153 505 560 51 566 452 53 524 49 194 42370 511 509 456 8 337 59 35 42360 455 475 40496 531 1063 703 534 1467 44968 42 1510 334 549 11 188 29 470 43611 40981 45102 1464 1549 37 43962 469 458 54 45545 50 307 1555 31 1494 4544 41702 934 1479 41021 41265 185 454 1462 43466 23 43919 42931 1501 1493 1492 1504 315 20 12 14 16 22 18 1067 1466 36 1487 44091 42727 43926 41143 507 46 3 44055 44061 1043 44160 44158 40900 44124 1489 1497 40499 24 41145 1475 182 44136 43986 503 372 44157 558 44132 562 189 40536 44056 422 44054 4538 45062 44145 44122 1531 1459 44126 44062 42183 32 4534 42734 44125 44123 44137 1476 44005 1471 44133 846 44134 44162 44089 44063 44026 45012 6 44090 537 999999 44148 44066 44984 4135 1486 45714 44064 344 41027 151 44963 40985 45068 44059 44131 40685 45548 41169 41162 42345 41168 40922 23512 40672 44161 41150 1509 44057 43928 44069 1503 44068 44159 1113 1169 150 44065 44129 1567"
    data_ids = datalist.split(" ")
    if args.openml_id is not None:
        if str(args.openml_id) not in data_ids:
            parser.error("openml_id is not in the official 196-dataset benchmark")
        data_ids = [str(args.openml_id)]
    large_set = "44059 44131 40685 45548 41169 41162 42345 41168 40922 23512 40672 44161 41150 1509 44057 43928 44069 1503 44068 44159 1113 44027 1169 150 44065 44129 1567"
    large_set = large_set.split(" ")

    for data_id in data_ids:
        tasktype = data_info[data_id]["tasktype"]
        seeds = 3 if data_id in large_set else 10
        for seed in (range(seeds) if args.seed is None else [args.seed]):
            print("==================", data_id, seed, "==================")
            dataset = TabularDataset(int(data_id), tasktype, device=device, seed=seed)
            for modelname in models:
                get_results(result_fname, dataset, tasktype=tasktype, seed=seed, data_id=data_id, model=modelname, ensemble_type="deep", basepath=basepath, allow_legacy_logs=args.allow_legacy_logs)
                get_results(result_fname, dataset, tasktype=tasktype, seed=seed, data_id=data_id, model=modelname, ensemble_type="hyper", basepath=basepath, allow_legacy_logs=args.allow_legacy_logs)
                get_results(result_fname, dataset, tasktype=tasktype, seed=seed, data_id=data_id, model=modelname, ensemble_type="all", basepath=basepath, allow_legacy_logs=args.allow_legacy_logs)

    print(result_fname)

if __name__ == "__main__":
    main()

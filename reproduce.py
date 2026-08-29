## Main file for reproducing performance with the optimal configuration for a given set of [algorithm, dataset, preprocessing method].
## Paper info: MultiTab: A Comprehensive Benchmark Suite with Multi-Dimensional Analysis in Tabular Domains
## Contact author: Kyungeun Lee (kyungeun.lee@lgresearch.ai)

import optuna, argparse, os, torch, json, joblib, time, datetime, sys, shutil, pickle
from libs.data import TabularDataset
from libs.model import *
from libs.eval import *
from libs.search_space import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import os
token = os.getenv("HF_TOKEN")
def is_study_todo(study, tasktype, optimal_value=1.0, num_trials=100):
    # Check if the study reached the optimal goal set in the callback
    if tasktype != "regression":
        if study.best_value >= optimal_value:
            # print(f"Study reached the optimal value of {optimal_value}.")
            return False

    # Check if the study has completed the minimum number of trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == num_trials:
        return False

    donelen = len(completed_trials)
    # print(donelen)
    return True

error_fname = "error.log"
errors = pd.DataFrame(columns=("seed", "data", "model"))
i = 0
# ⚠ 원본은 try 없이 열어서 error.log 가 없으면 모듈 로드 단계에서 죽는다.
#   이 파일은 "HPO 가 실패했던 조합"을 스킵하기 위한 것이므로, 없으면
#   스킵할 조합이 없다는 뜻이고 그대로 진행하면 된다.
try:
    _ef = open(error_fname, "r")
except FileNotFoundError:
    _ef = None
if _ef is not None:
  with _ef as file:
    for line in file:
        l = line.split(f"optim_logs{os.sep}")[-1]
        seed = l.split("seed=")[-1].split(os.sep)[0]
        data = l.split("data=")[-1].split("..")[0]
        model = l.split("model=")[-1].split(".pkl")[0]
        errors.loc[i] = [seed, data, model]
        i += 1

# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments to the parser for GPU ID, OpenML dataset ID, code directory, model name, preprocessing method, and categorical feature threshold
parser.add_argument("--gpu_id", type=int, default=4)
parser.add_argument("--openml_id", type=int, default=10)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--savepath", type=str, default=".", help="path to save the results")

# Parse the arguments
args = parser.parse_args()

# Load dataset information from a JSON file
with open(f'./dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info.get(str(args.openml_id))['tasktype']

# directory = os.path.join(args.savepath, f'reproduce_logs/seed={args.seed}/data={args.openml_id}')
directory = os.path.join(args.savepath, 'reproduce_logs', f'seed={args.seed}', f'data={args.openml_id}')
if not os.path.exists(directory):
    os.makedirs(directory)

# MultiTab 논문 Table 2 의 13개 모델. 아래는 의도적으로 제외한다:
#   lr, tabpfn : 비교 대상이 아니다. tabpfn 은 X_train>3000 에서 sys.exit() 로
#                프로세스를 죽여 뒤따르는 모델까지 못 돌게 만든다.
#   ptarl      : 논문에 없는 추가 모델이고 HPO 도 불완전하다.
#   tabm       : 논문에 없는 추가 모델. MultiTab 프로토콜에 맞춰 튜닝된 상태가
#                아니므로 벤치마크 비교표에서 뺀다.
models = ["randomforest", "xgboost", "catboost", "lightgbm",
          "mlp", "embedmlp", "mlpplr", "resnet",
          "ftt", "t2gformer", "saint", "tabr", "modernnca"]

# (init_hp, deepens, hyperens)
opts = [(True, 0, 0), #no HPO
        (False, 0, 0), #tuned
        (False, 1, 0), (False, 2, 0), (False, 3, 0), (False, 4, 0), #deep ensemble
        (False, 0, 1), (False, 0, 2), (False, 0, 3), (False, 0, 4)] #hyper ensemble

# ⚠ tuned 하나만 쓴다. 논문 §3.3 은 "retrain the model using the best
#   configuration found on that fold's validation split" 이고, deep/hyper
#   ensemble 은 본 논문 분석(Table 2)에 등장하지 않는다.
#   opts 전체를 돌리면 (dataset, seed) 당 88 회 학습이 되어 25 x 5 기준
#   11,000 회가 된다. tuned 만이면 13 x 125 = 1,625 회다.
opt_dict = {m: [opts[1]] for m in models}

# Set GPU environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# torch.cuda.set_device(args.gpu_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import platform
env_info = '{0}:{1}'.format(platform.node(), args.gpu_id)

# Load dataset with specified preprocessing
dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)

# Split dataset into training, validation, and test sets
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
y_std = dataset.y_std

for m in models:
    for (init_hps, ensemble_deep, ensemble_hyper) in opt_dict[m]:
        fname = os.path.join(directory, f'model={m}..init_hps={init_hps}..deep={ensemble_deep}..hyper={ensemble_hyper}.npy')
        todo = (os.path.exists(fname) == False)
        print("##########################################")
        print(env_info) 
        print(fname)
        print(todo)
        print("##########################################")
        if todo:
            
            params = {}
            if m not in ["tabpfn", "lr"]:
                # Load the optimization logs
                try:
                    # opt_logs = joblib.load(os.path.join(args.savepath, f'optim_logs/seed={args.seed}/data={args.openml_id}..model={m}.pkl'))
                    opt_logs = joblib.load(os.path.join(args.savepath, 'optim_logs', f'seed={args.seed}', f'data={args.openml_id}..model={m}.pkl'))
                    not_complete = is_study_todo(opt_logs, tasktype)
                    assert not_complete == False
                except FileNotFoundError:
                    if len(errors[(errors["seed"] == str(args.seed)) & (errors["model"] == m) & (errors["data"] == str(args.openml_id))]) > 0:
                        np.save(fname, "ValueError: Not implemented.")
                        continue
                    else:
                        print("Not Yet")
                        continue
                        
                if init_hps:
                    params = opt_logs.trials[0].params
                elif ensemble_hyper > 0:
                    completed_trials = [trial for trial in opt_logs.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                    if len(completed_trials) <= ensemble_hyper:
                        np.save(fname, "ValueError: Not implemented.")
                        continue
                    assert len(completed_trials) > ensemble_hyper
                    if tasktype == "regression":
                        sorted_trials = sorted(completed_trials, key=lambda x: x.value)
                    else:
                        sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)
                    params = sorted_trials[ensemble_hyper].params
                else:
                    params = opt_logs.best_params
                # Add default(fixed) parameters
                params = add_default_params(m, params, args.openml_id)

            params = rearrange_params(m, args.openml_id, params)
            
            # Check for class imbalance problems in multiclass tasks with specific models
            if (tasktype == "multiclass") & (m in ["catboost", "xgboost", "lightgbm"]):
                if y_train.size(1) != y_train.unique(dim=1).size(1):            
                    raise ValueError # "Unknown class problem" --- Inherent challenges in GBDTs
            
            # Define and train the model
            output_dim = y_train.shape[1] if tasktype == "multiclass" else 1
            if m in ["tabpfn", "lr"]:
                model = getmodel(m, {}, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
            else:
                model = getmodel(m, params, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
        
            st = time.time()
            try:
                model.fit(X_train, y_train, X_val, y_val)
                et = time.time()
            except ValueError:
                # ⚠ 원본은 sys.exit() 였다. 한 모델의 학습 실패가 그
                #   (dataset, seed) 의 남은 모델까지 전부 못 돌게 만든다.
                np.save(fname, "ValueError: Not implemented.")
                continue
        
            # Model inference
            preds_test = model.predict(X_test)
            # For classification tasks with ensemble techniques, we should calculate probability or logits
            preds_test_prob = model.predict_proba(X_test, logit=True) if tasktype != "regression" else None
            inference_results = {"Prediction": preds_test, "Probability": preds_test_prob, "time": et - st}
            
            if tasktype == "regression":
                test_metrics = calculate_metric(y_test*y_std, preds_test*y_std, None, tasktype, 'test')
            else:
                test_metrics = calculate_metric(y_test, preds_test, preds_test_prob, tasktype, 'test')
            inference_results["Performance"] = test_metrics

            # print(inference_results["Prediction"])
            print(device, env_info, args.openml_id, data_info.get(str(args.openml_id))['name'], m, args.savepath)
            print(test_metrics)
        
            print("#############################################")
            np.save(fname, inference_results)
            print("#############################################")
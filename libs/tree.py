from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

class LR(torch.nn.Module):
    def __init__(self, tasktype):
        self.tasktype = tasktype
        self.model = LinearRegression() if self.tasktype == "regression" else LogisticRegression()
        
    def fit(self, X_train, y_train, X_val, y_val):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())
    
    def predict_proba(self, X_test, logit=False):
        probs = self.model.predict_proba(X_test.cpu().numpy())
        if logit:
            # Prevent division by zero in logit calculation
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            if probs.shape[1] == 2:  # Binary classification
                logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                return logits
            else:  # Multiclass classification
                # ⚠ softmax 의 역변환은 log(p) 다. per-class log-odds
                #   log(p/(1-p)) 를 쓰면 eval.py 가 softmax 를 적용했을 때
                #   원래 확률이 복원되지 않고 과도하게 sharpen 된다. 실측:
                #     p = [0.70, 0.20, 0.10]
                #       log(p/(1-p)) -> [0.866, 0.093, 0.041]  log loss 0.3851 -> 0.2245 (-42%)
                #       log(p)       -> [0.700, 0.200, 0.100]  log loss 정확히 일치
                #   softmax(log p) = p 이므로 상수배를 제외하고 정확한 역변환이다.
                #
                # ⚠ 위 이진 분기의 log(p/(1-p)) 는 eval.py 가 expit 를 적용해
                #   정확히 복원되므로 그대로 둔다. 이진과 다중의 역함수가
                #   다르기 때문이다(expit vs softmax).
                logits = np.log(probs)
                return logits
        else:
            return probs

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class KNN(torch.nn.Module):
    def __init__(self, tasktype, n_neighbors=5):
        super(KNN, self).__init__()
        self.tasktype = tasktype
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors) if self.tasktype == "regression" else KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())

    def predict_proba(self, X_test, logit=False):
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_test.cpu().numpy())
            if logit:
                # Prevent division by zero in logit calculation
                probs = np.clip(probs, 1e-9, 1 - 1e-9)
                if probs.shape[1] == 2:  # Binary classification
                    logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                    return logits
                else:  # Multiclass classification
                    # softmax의 역변환은 log(p) (LR 클래스의 주석 참고)
                    logits = np.log(probs)
                    return logits
            else:
                return probs
        else:
            raise NotImplementedError("Probability predictions are not supported for regression tasks.")

class DecisionTree(torch.nn.Module):
    def __init__(self, tasktype):
        super(DecisionTree, self).__init__()
        self.tasktype = tasktype
        self.model = DecisionTreeRegressor() if self.tasktype == "regression" else DecisionTreeClassifier()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())

    def predict_proba(self, X_test, logit=False):
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_test.cpu().numpy())
            if logit:
                # Prevent division by zero in logit calculation
                probs = np.clip(probs, 1e-9, 1 - 1e-9)
                if probs.shape[1] == 2:  # Binary classification
                    logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                    return logits
                else:  # Multiclass classification
                    # softmax의 역변환은 log(p) (LR 클래스의 주석 참고)
                    logits = np.log(probs)
                    return logits
            else:
                return probs
        else:
            raise NotImplementedError("Probability predictions are not supported for regression tasks.")


class RandomForest(torch.nn.Module):
    def __init__(self, params, tasktype):
        self.params = params
        self.tasktype = tasktype
        if self.tasktype == "regression":
            self.model = RandomForestRegressor(**params, n_jobs=-1)
        else:
            self.model = RandomForestClassifier(**params, n_jobs=-1)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tasktype == "multiclass":
            y_train = torch.argmax(y_train, dim=1)
        self.model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    
    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())
    
    def predict_proba(self, X_test, logit=False):
        probs = self.model.predict_proba(X_test.cpu().numpy())
        if logit:
            # Prevent division by zero in logit calculation
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            if probs.shape[1] == 2:  # Binary classification
                logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                return logits
            else:  # Multiclass classification
                # softmax의 역변환은 log(p) (LR 클래스의 주석 참고).
                # log(p/(1-p))를 쓰면 eval.py의 softmax가 원래 확률을 복원하지
                # 못하고 과도하게 sharpen 되어 logloss/AUROC가 왜곡된다.
                logits = np.log(probs)
                return logits
        else:
            return probs

class CatBoost(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "MultiClass", "binclass": "CrossEntropy", "regression": "RMSE"}
        eval_fn = {"multiclass": "Accuracy", "binclass": "Accuracy", "regression": "RMSE"}
        model_fn = {"multiclass": CatBoostClassifier, "binclass": CatBoostClassifier, "regression": CatBoostRegressor}
            
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.model = model_fn[tasktype](loss_function=loss_fn[tasktype], eval_metric=eval_fn[tasktype], cat_features=cat_features, **params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
        
        X_train = pd.DataFrame(X_train.cpu()).astype({k: 'int' for k in self.cat_features})
        y_train = np.argmax(y_train.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_train.cpu().numpy()
                
        X_val = pd.DataFrame(X_val.cpu()).astype({k: 'int' for k in self.cat_features})
        y_val = np.argmax(y_val.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_val.cpu().numpy()
         
        dtrain = Pool(X_train, label=y_train, cat_features=self.cat_features)
        dval = Pool(X_val, label=y_val, cat_features=self.cat_features)
        
        self.model.fit(dtrain, eval_set=dval, use_best_model=True, verbose=0)
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        preds = self.model.predict(X_test)
        # multiclass: CatBoost가 float 2D 또는 문자열로 반환하는 경우 정수 1D로 변환
        if self.tasktype == "multiclass":
            return np.array(preds).flatten().astype(int)
        return preds

    def predict_proba(self, X_test, logit=False):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        if logit:
            return self.model.predict(X_test, prediction_type='RawFormulaVal')
        else:
            return self.model.predict_proba(X_test)

    
class XGBoost(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "multi:softmax", "binclass": "binary:logistic", "regression": "reg:squarederror"}
        model_fn = {"multiclass": XGBClassifier, "binclass": XGBClassifier, "regression": XGBRegressor}
            
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.model = model_fn[tasktype](
            booster='gbtree', tree_method='hist', objective=loss_fn[tasktype], **params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
    
        X_train = pd.DataFrame(X_train.cpu()).astype({k: 'int' for k in self.cat_features})
        y_train = np.argmax(y_train.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_train.cpu().numpy()
        
        X_val = pd.DataFrame(X_val.cpu()).astype({k: 'int' for k in self.cat_features})
        y_val = np.argmax(y_val.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_val.cpu().numpy()
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        preds = self.model.predict(X_test)
        # multiclass: XGBoost multi:softmax가 float로 반환하는 경우 정수 1D로 변환
        if self.tasktype == "multiclass":
            return np.array(preds).flatten().astype(int)
        return preds
    
    def predict_proba(self, X_test, logit=False):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        if logit:
            return self.model.predict(X_test, output_margin=True)
        else:
            return self.model.predict_proba(X_test)

    
class LightGBM(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "multiclass", "binclass": "binary", "regression": "regression"}
        model_fn = {"multiclass": LGBMClassifier, "binclass": LGBMClassifier, "regression": LGBMRegressor}
            
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.model = model_fn[tasktype](objective=loss_fn[tasktype], **params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
        
        X_train = pd.DataFrame(X_train.cpu()).astype({k: 'category' for k in self.cat_features})
        y_train = np.argmax(y_train.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_train.cpu().numpy()
        
        X_val = pd.DataFrame(X_val.cpu()).astype({k: 'category' for k in self.cat_features})
        y_val = np.argmax(y_val.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_val.cpu().numpy()
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=None, categorical_feature="auto")
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'category' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test, logit=False):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'category' for k in self.cat_features})
        if logit:
            return self.model.predict(X_test, raw_score=True)
        else:
            return self.model.predict_proba(X_test)
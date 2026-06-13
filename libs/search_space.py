import optuna
import numpy as np

large_datalist = [44159, 1113, 44027, 41960, 1169, 150, 44065, 44129, 1567, 5, 20, 12, 41147, 422]
### Define hyperparameter search space (Supplementary E)
def get_search_space(trial, modelname, num_features=None, data_id=None, metric=None):
    if modelname == "randomforest":
        assert num_features is not None
        params = {
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5000, 50000),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 3, 4, 5, 10, 20, 40, 80]),
            'max_features': trial.suggest_categorical('max_features', [int(np.sqrt(num_features)), int(np.log2(num_features)), 0.5, 0.75, 1.0]),
            'n_estimators': 300
        }
    elif modelname == "xgboost":
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 1.5),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.1, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'enable_categorical': trial.suggest_categorical('enable_category', [True, False]),
            'early_stopping_rounds': 20,
            'n_estimators': 10000,
            'max_iterations': 10000,
            'verbosity': 0
        }
    elif modelname == "catboost":
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.1, log=True),
            'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 5),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
            'grow_policy': trial.suggest_categorical('grow_policy', ["SymmetricTree", "Depthwise"]),
            'early_stopping_rounds': 20,
            'iterations': 10000,
            'verbose': 0,
            'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [2, 3, 5, 10]),
        }
    elif modelname == "lightgbm":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 60),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'early_stopping_rounds': 20,
            'iterations': 10000,
            'verbosity': -1
        }
    elif modelname == "mlp":
        params = {
            'depth': trial.suggest_int('depth', 1, 8),
            'width': trial.suggest_int('width', 1, 512),
            'dropout': trial.suggest_float('dropout', 0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'normalization': trial.suggest_categorical('normalization', [None, "batchnorm", "layernorm"]), 
            'activation': trial.suggest_categorical('activation', ["relu", "lrelu", "sigmoid", "tanh", "gelu"]), 
            'n_epochs': 100,
            'early_stopping_rounds': 20,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "embedmlp":
        params = {
            'depth': trial.suggest_int('depth', 1, 8),
            'width': trial.suggest_int('width', 1, 512),
            'd_embedding': trial.suggest_int('d_embedding', 64, 512),
            'dropout': trial.suggest_float('dropout', 0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'normalization': trial.suggest_categorical('normalization', [None, "batchnorm", "layernorm"]), 
            'activation': trial.suggest_categorical('activation', ["relu", "lrelu", "sigmoid", "tanh", "gelu"]), 
            'n_epochs': 100,
            'early_stopping_rounds': 20,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "mlpplr":
        params = {
            'depth': trial.suggest_int('depth', 1, 8),
            'width': trial.suggest_int('width', 1, 512),
            'd_embedding_cat': trial.suggest_int('d_embedding_cat', 64, 512),
            'd_embedding_num': trial.suggest_int('d_embedding_num', 1, 128),
            'dropout': trial.suggest_float('dropout', 0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'normalization': trial.suggest_categorical('normalization', [None, "batchnorm", "layernorm"]), 
            'activation': trial.suggest_categorical('activation', ["relu", "lrelu", "sigmoid", "tanh", "gelu"]), 
            'n_epochs': 100,
            'early_stopping_rounds': 20,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "resnet":
        params = {
            'n_layers': trial.suggest_int('n_layers', 1, 8),
            'd': trial.suggest_int('d', 64, 512),
            'd_embedding': trial.suggest_int('d_embedding', 64, 512),
            'd_hidden_factor': trial.suggest_float('d_hidden_factor', 1, 4),
            'hidden_dropout': trial.suggest_float('hidden_dropout', 0, 0.5),
            'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.5),
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            'normalization': trial.suggest_categorical('normalization', [None, "batchnorm", "layernorm"]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'n_epochs': 100,
            'early_stopping_rounds': 20,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "ftt":
        params = {
            ## tokenizer
            'token_bias': trial.suggest_categorical('token_bias', [True, False]),
            ## transformer
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'd_token': trial.suggest_int('d_token', 8, 64), ##original: trial.suggest_int('d_token', 64, 512), -> it should be divided into 8(=n_heads)
            'n_heads': 8, #do not tune in original paper
            'd_ffn_factor': trial.suggest_float('d_ffn_factor', 2/3, 8/3),
            'attention_dropout': trial.suggest_float('attention_dropout', 0, 0.5),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0, 0.5),
            'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.2),
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            'prenormalization': trial.suggest_categorical('prenormalization', [True, False]),
            'initialization': trial.suggest_categorical('initialization', ["xavier", "kaiming"]),
            ## linformer
            'kv_compression': None, ## default setup in original paper
            'kv_compression_sharing': None, ## default setup in original paper
            ## optimizer
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'n_epochs': 100,
            'early_stopping_rounds': 20,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "t2gformer":
        params = {
            'n_layers': trial.suggest_int('n_layers', 1, 5),
            'd_token': trial.suggest_int('d_token', 8, 64), ##original: trial.suggest_int('d_token', 64, 512), -> it should be divided into 8(=n_heads) -- same as ftt
            'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.2),
            'attention_dropout': trial.suggest_float('attention_dropout', 0, 0.5),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'learning_rate_embed': trial.suggest_float('learning_rate_embed', 5e-3, 5e-2, log=True),
            'n_heads': 8, ## default setup in original paper
            'token_bias': True, ## default setup in original paper
            'kv_compression': None, ## default setup in original paper
            'kv_compression_sharing': None, ## default setup in original paper
            'd_ffn_factor': trial.suggest_float('d_ffn_factor', 2/3, 8/3),
            'prenormalization': trial.suggest_categorical('prenormalization', [True, False]),
            'initialization': trial.suggest_categorical('initialization', ["xavier", "kaiming"]),
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            'early_stopping_rounds': 20,
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3), 
            'n_epochs': 100,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "saint":
        params = {
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            'depth' : 3 if data_id in large_datalist else 6,
            'heads' : 4 if data_id in large_datalist else 8,
            'hidden' : 16,
            'attn_dropout': trial.suggest_float('attn_dropout', 0, 0.3),
            'ff_dropout': trial.suggest_float('ff_dropout', 0, 0.8),
            # 'cont_embeddings': trial.suggest_categorical('cont_embeddings', ['MLP','Noemb','pos_singleMLP']),
            'cont_embeddings': 'MLP',
            'attentiontype':'colrow',
            'final_mlp_style': trial.suggest_categorical('final_mlp_style', ['common', 'sep']),
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2), 
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'early_stopping_rounds': 20,
            'n_epochs': 100
        }
        if data_id in [5, 1486, 1501, 20, 12, 41143, 44061, 1476, 41702, 41145, 41147, 422]:
            params["embedding_dim"] = 8
    elif modelname == "modernnca":
        large_set = [
            44059, 44131, 40685, 45548, 41169, 41162, 42345, 41168, 40922, 23512, 40672, 44161, 41150, 1509, 44057, 
            43928, 44069, 1503, 44068, 44159, 1113, 44027, 1169, 150, 44065, 44129, 1567]
        params = {
            "model": {
            "d_block": trial.suggest_int('d_block', 64, 128) if data_id in large_set else trial.suggest_int('d_block', 64, 1024),
            "dim": trial.suggest_int('dim', 64, 128) if data_id in large_set else trial.suggest_int('dim', 64, 1024),
            "dropout": trial.suggest_float('dropout', 0, 0.5),
            "n_blocks": 0 if data_id in large_set else trial.suggest_int('n_blocks', 0, 2),
            "num_embeddings": {            
                "d_embedding": trial.suggest_int('d_embedding', 8, 32) if data_id in large_set else trial.suggest_int('d_embedding', 16, 64),
                "frequency_scale": trial.suggest_float('frequency_scale', 0.005, 10.0, log=True), 
                "n_frequencies": trial.suggest_int('n_frequencies', 16, 96)}},
            "lr": trial.suggest_float('lr', 1e-05, 0.1, log=True), 
            "weight_decay": trial.suggest_float('weight_decay', 1e-06, 0.001, log=True), 'early_stopping_rounds': 20, 'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False])
        }
    elif modelname == "tabr":
        large_set = [
            44059, 44131, 40685, 45548, 41169, 41162, 42345, 41168, 40922, 23512, 40672, 44161, 41150, 1509, 44057, 
            43928, 44069, 1503, 44068, 44159, 1113, 44027, 1169, 150, 44065, 44129, 1567]
        params = {
            "model": {
                "d_main": trial.suggest_int('d_main', 96, 384),
                "context_dropout": trial.suggest_float('context_dropout', 0.0, 0.6),
                "encoder_n_blocks": trial.suggest_int('encoder_n_blocks', 0, 1),
                "predictor_n_blocks": trial.suggest_int('predictor_n_blocks', 1, 2),
                "dropout0": trial.suggest_float('dropout0', 0.0, 0.6),
                "d_multiplier": 2.0, "mixer_normalization": "auto", "dropout1": 0.0, "normalization": "LayerNorm", "activation": "ReLU",
                "feature_interaction" : True,
                "num_embeddings": {            
                    "d_embedding": trial.suggest_int('d_embedding', 8, 32) if data_id in large_set else trial.suggest_int('d_embedding', 16, 64),
                    "frequency_scale": trial.suggest_float('frequency_scale', 0.01, 100.0, log=True), 
                    # "n_frequencies": trial.suggest_int('n_frequencies', 16, 96), # PLREncoding용
                    "n_bins": 32,
                    "d_embedding" : 64,
                }, 
                # "metric": metric if metric is not None else 'l2'  # 인자로 받은 metric을 그대로 사용하거나, 없으면 기본값 'l2'로 설정
                "metric" : "l2",  # 'l2', 'l1', 'cosine', 'mahalanobis', 'wasserstein', 'kl'
                "lambda_w": trial.suggest_float('lambda_w', 0.01, 0.5, log=True),
                "margin": trial.suggest_float('margin', 0.5, 2.0),
            },
            "lr": 1e-4,
            "weight_decay": 1e-5,
            # "weight_decay": trial.suggest_float('weight_decay', 1e-06, 0.001, log=True),
            "early_stopping_rounds": 10,
            'lr_scheduler': True,
        }
    elif modelname == "ptarl":
        # PTaRL: Prototype-based Tabular Representation Learning via Space Calibration
        # Hangting Ye et al., ICLR 2024 — https://arxiv.org/abs/2407.05364
        # 원본 코드(HangtingYe/PTaRL: models.py, train_final_version.py) 구조에 충실한 버전.
        #
        #   - encoder: vanilla MLP (Linear-ReLU-Dropout) — d_hidden, n_blocks로 d_layers 구성
        #   - lr: 1e-4 고정 (원본 하드코딩, 탐색하지 않음) → libs/ptarl.py에서 처리
        #   - n_epochs: early stopping(patience=20)만으로 종료, max_epochs는 안전 상한
        #   - ot_weight / diversity_weight / r_weight: 원본처럼 3개 독립 가중치
        #     (원본 default=0.25, 여기서는 Optuna로 각각 탐색)
        #   - n_clusters = ceil(log2(n_num+n_cat)) — libs/ptarl.py에서 자동 계산 (탐색 X)
        large_set = [
            44059, 44131, 40685, 45548, 41169, 41162, 42345, 41168, 40922, 23512, 40672,
            44161, 41150, 1509, 44057, 43928, 44069, 1503, 44068, 44159, 1113, 44027,
            1169, 150, 44065, 44129, 1567,
        ]
        params = {
            "d_hidden":         trial.suggest_categorical('d_hidden', [64, 128])
                                if data_id in large_set
                                else trial.suggest_categorical('d_hidden', [64, 128, 256]),
            "n_blocks":         trial.suggest_int('n_blocks', 1, 4),
            "dropout":          trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
            "ot_weight":        trial.suggest_float('ot_weight', 0.05, 1.0, log=True),
            "diversity_weight": trial.suggest_float('diversity_weight', 0.05, 1.0, log=True),
            "r_weight":         trial.suggest_float('r_weight', 0.05, 1.0, log=True),
            "weight_decay":     trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            "max_epochs":            100,  # 다른 baseline(mlp/ftt/resnet 등)과 동일한 예산
            "early_stopping_rounds": 20,  # 원본 고정값 (patience=20)
        }
    elif modelname == "tabm":
        # TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling
        # Gorishniy et al., ICLR 2025 — https://arxiv.org/abs/2410.24210
        # k=32 고정 (논문 §3.3), tune: d_block, n_blocks, dropout, lr, weight_decay
        large_set = [
            44059, 44131, 40685, 45548, 41169, 41162, 42345, 41168, 40922, 23512, 40672,
            44161, 41150, 1509, 44057, 43928, 44069, 1503, 44068, 44159, 1113, 44027,
            1169, 150, 44065, 44129, 1567,
        ]
        params = {
            "k":            32,
            "d_block":      trial.suggest_categorical('d_block', [64, 128, 256])
                            if data_id in large_set
                            else trial.suggest_categorical('d_block', [128, 256, 512]),
            "n_blocks":     trial.suggest_int('n_blocks', 1, 4),
            "dropout":      trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
            "lr":           trial.suggest_float('lr',           1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            "lr_scheduler": trial.suggest_categorical('lr_scheduler', [True, False]),
            "n_epochs":              100,
            "early_stopping_rounds": 20,
        }
    return params


def add_default_params(modelname, params, data_id):
    """
    이미 정의된 data_id 체계를 활용하여 기본 파라미터를 동적으로 추가합니다.
    """
    if modelname == "xgboost":
        params.update({
            'n_estimators': 10000,
            'early_stopping_rounds': 20,
            'verbosity': 0
        })
    elif modelname == "lightgbm":
        params.update({
            'iterations': 10000,
            'early_stopping_rounds': 20,
            'verbosity': -1
        })
    elif modelname in ["mlp", "resnet", "ftt", "embedmlp", "mlpplr", "saint", "t2gformer"]:
        # 데이터 크기에 따라 학습 횟수를 조절하는 소프트코딩
        params.update({
            'n_epochs': 50 if data_id in large_datalist else 100,
            'early_stopping_rounds': 20
        })
    elif modelname in ["ptarl", "tabm"]:
        # 모든 필요 파라미터가 get_search_space에 포함되어 있음
        pass
    return params


def rearrange_params(modelname, data_id, params):
    """
    Optuna 로그에 누락된 고정 파라미터들을 데이터 ID에 맞춰 복원합니다.
    """
    # 1. FT-Transformer & T2G-Former 보정
    if modelname in ['ftt', 't2gformer']:
        params.setdefault('kv_compression', None)
        params.setdefault('kv_compression_sharing', None)
        params.setdefault('n_heads', 8)

    # 2. SAINT 모델 보정 (사용자님의 로직 반영)
    elif modelname == "saint":
        # large_datalist는 search_space.py 상단에 이미 정의되어 있어야 합니다.
        if 'depth' not in params:
            params['depth'] = 3 if data_id in large_datalist else 6
        if 'heads' not in params:
            params['heads'] = 4 if data_id in large_datalist else 8
        if 'hidden' not in params:
            params['hidden'] = 16
        if 'cont_embeddings' not in params:
            params['cont_embeddings'] = 'MLP'
        if 'attentiontype' not in params:
            params['attentiontype'] = 'colrow'
        
        # 특정 ID에 대한 임베딩 차원 보정
        if data_id in [5, 1486, 1501, 20, 12, 41143, 44061, 1476, 41702, 41145, 41147, 422]:
            params.setdefault("embedding_dim", 8)
        else:
            params.setdefault("embedding_dim", 32) # 기본값 보장

    # 3. TabR & ModernNCA 보정 (d_multiplier 등 고정값)
    elif modelname == "tabr":
        params.setdefault("model", {})
        if isinstance(params["model"], dict):
            params["model"].setdefault("d_multiplier", 2.0)
            params["model"].setdefault("mixer_normalization", "auto")
            params["model"].setdefault("dropout1", 0.0)
            params["model"].setdefault("normalization", "LayerNorm")
            params["model"].setdefault("activation", "ReLU")

    # 4. PTaRL 보정
    elif modelname == "ptarl":
        large_set = [
            44059, 44131, 40685, 45548, 41169, 41162, 42345, 41168, 40922, 23512, 40672,
            44161, 41150, 1509, 44057, 43928, 44069, 1503, 44068, 44159, 1113, 44027,
            1169, 150, 44065, 44129, 1567,
        ]
        params.setdefault("d_hidden",  128)
        params.setdefault("n_blocks",  1)
        params.setdefault("dropout",   0.1)
        params.setdefault("ot_weight",        0.25)  # 원본 default
        params.setdefault("diversity_weight", 0.25)  # 원본 default
        params.setdefault("r_weight",         0.25)  # 원본 default
        params.setdefault("weight_decay", 1e-5)
        params.setdefault("max_epochs", 100)
        params.setdefault("early_stopping_rounds", 20)

    # 5. TabM 보정
    elif modelname == "tabm":
        params.setdefault("k",                     32)
        params.setdefault("n_epochs",              100)
        params.setdefault("early_stopping_rounds", 20)
        params.setdefault("lr_scheduler",          True)

    return params


def suggest_initial_trial(modelname):
    init_values = {
        "randomforest": {}, # TabRepo
        "xgboost": {"max_depth": 6, "min_child_weight": 1.0, "colsample_bytree": 1.0}, # TabRepo
        "catboost": {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3, "max_ctr_complexity": 4}, # TabRepo
        "lightgbm": {"learning_rate": 0.05, "feature_fraction": 1.0, "min_data_in_leaf": 20, "num_leaves": 31}, # TabRepo
        "mlp": {"learning_rate": 3e-4, "weight_decay" : 1e-6, "dropout": 0.1, "depth": 2, "width": 128, "activation": "relu", "optimizer": "AdamW"}, # TabRepo, Gorishniy 
        "embedmlp": {"learning_rate": 3e-4, "weight_decay" : 1e-6, "dropout": 0.1, "depth": 2, "width": 128, "activation": "relu", "optimizer": "AdamW"}, # TabRepo, Gorishniy 
        "mlpplr": {"learning_rate": 3e-4, "weight_decay" : 1e-6, "dropout": 0.1, "depth": 2, "width": 128, "activation": "relu", "optimizer": "AdamW", "d_embedding_num": 8}, # TabRepo, Gorishniy 
        "resnet": {"learning_rate": 3e-4, "weight_decay" : 1e-6, "activation": "relu", "optimizer": "AdamW", "normalization": "batchnorm"}, # TabRepo, Gorishniy 
        "ftt": {"learning_rate": 1e-4, "weight_decay" : 1e-5, "optimizer": "AdamW", "n_layers": 3, "d_token": 24, 'n_heads': 8, 'activation': "reglu", "d_ffn_factor": 4/3, 
                "attention_dropout": 0.2, "ffn_dropout": 0.1, "residual_dropout": 0, "initialization": "kaiming"}, # Gorishniy
        "t2gformer": {"activation": "relu", "optimizer": "AdamW"}, # t2gformer
        "saint": {"learning_rate": 0.0001, "weight_decay" : 0.01, "activation": "relu", "optimizer": "AdamW", "attn_dropout": 0.1, "ff_dropout": 0.8}, #SAINT
        "modernnca": {"n_blocks": 0, "weight_decay": 0.0002, "lr": 0.01},
        "tabr": {"d_main": 265, "encoder_n_blocks": 0, "predictor_n_blocks": 1},
        # PTaRL 원본 default (train_final_version.py 의 CLI default 와 동일: 모두 0.25)
        "ptarl": {
            "d_hidden":         128,
            "n_blocks":         1,
            "dropout":          0.1,
            "ot_weight":        0.25,
            "diversity_weight": 0.25,
            "r_weight":         0.25,
            "weight_decay":     1e-5,
        },
        # TabM 논문 README default (lr=2e-3, weight_decay=3e-4, k=32)
        "tabm": {
            "k":                     32,
            "d_block":               256,
            "n_blocks":              2,
            "dropout":               0.1,
            "lr":                    2e-3,
            "weight_decay":          3e-4,
            "lr_scheduler":          True,
            "n_epochs":              100,
            "early_stopping_rounds": 20,
        },
    }
    assert modelname in init_values
    return init_values[modelname]
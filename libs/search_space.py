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
            # ⚠ 'iterations'는 LightGBM의 유효한 alias가 아니다 (num_iterations의
            #   alias 목록: num_iteration, n_iter, num_trees, n_estimators 등).
            #   'iterations': 10000 으로 넘기면 조용히 무시되어 sklearn 기본값
            #   100 그루만 학습된다 (공식 MultiTab의 버그). n_estimators로 교체.
            'n_estimators': 10000,
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
                # 공식 MultiTab과 동일한 num_embeddings 탐색 공간.
                # (이전 버전: d_embedding 키가 중복되어 리터럴 64가 suggest 값을 덮었고,
                #  lambda_w/margin/feature_interaction/metric/n_bins 는 TabR 구현에
                #  존재하지 않는 죽은 파라미터라 TPE 탐색 차원만 낭비했음 → 제거)
                "num_embeddings": {
                    "d_embedding": trial.suggest_int('d_embedding', 8, 32) if data_id in large_set else trial.suggest_int('d_embedding', 16, 64),
                    "frequency_scale": trial.suggest_float('frequency_scale', 0.01, 100.0, log=True),
                    "n_frequencies": trial.suggest_int('n_frequencies', 16, 96),
                },
            },
            "lr": 1e-4,
            "weight_decay": 1e-5,
            # "weight_decay": trial.suggest_float('weight_decay', 1e-06, 0.001, log=True),
            "early_stopping_rounds": 10,
            'lr_scheduler': True,
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
    """get_search_space() 가 리터럴로 넣는 값 중 best_params 에 남지 않는 것을 복원한다.

    ⚠ trial.suggest_* 를 거치지 않고 dict 에 직접 써넣은 값은 study.best_params
      에 없다. 여기서 되돌려 놓지 않으면 HPO 와 reproduce 가 서로 다른 조건으로
      학습한다 -- optimize.py 는 n_estimators=300 으로 탐색했는데 reproduce.py 는
      sklearn 기본값 100 으로 재학습하는 식이다.

    ⚠ libs/model.py 에도 같은 이름의 함수가 있다. reproduce.py 는
      `from libs.model import *` 다음에 `from libs.search_space import *` 를
      하므로 **이 정의가 이긴다**. model.py 쪽의 randomforest / catboost 분기가
      그래서 가려져 있었고, 두 모델이 조기종료도 없이 기본값으로 재학습되고
      있었다.

    ⚠ lr_scheduler 는 전 모델에서 trial.suggest_categorical 로 탐색되므로
      best_params 에 남는다. 여기서 건드리면 안 된다.

    [2026-08 감사] get_search_space() 의 리터럴 키를 자동 대조해 복원 누락을
    확인했다. 누락되어 있던 것:
        randomforest  n_estimators
        catboost      iterations, early_stopping_rounds, verbose
        t2gformer     token_bias
        tabr          model 중첩 구조 전체 + lr / weight_decay /
                      early_stopping_rounds(10) / lr_scheduler
        modernnca     model 중첩 구조 전체 + early_stopping_rounds(20)
                      (두 모델은 rearrange_params 가 처리한다)

    [2026-09 감사] 추가 수정:
        lightgbm      'iterations'는 유효 alias가 아니어서 무시됨 → n_estimators
        xgboost       optuna 이름 'enable_category' → 생성자 키워드
                      'enable_categorical' 번역 (reproduce 시 탐색값 유실 방지),
                      죽은 키 max_iterations 제거
    """
    if modelname == "randomforest":
        params.update({'n_estimators': 300})

    elif modelname == "xgboost":
        params.update({
            'n_estimators': 10000,
            'early_stopping_rounds': 20,
            'verbosity': 0,
        })
        # ⚠ optuna 파라미터 이름은 'enable_category'인데 XGBoost 생성자 키워드는
        #   'enable_categorical'이다. best_params 에는 전자로 남으므로 여기서
        #   번역해 주지 않으면 reproduce 시 탐색된 값이 조용히 버려지고
        #   기본값(False)으로 학습된다.
        if 'enable_category' in params:
            params['enable_categorical'] = params.pop('enable_category')

    elif modelname == "catboost":
        params.update({
            'iterations': 10000,
            'early_stopping_rounds': 20,
            'verbose': 0,
        })

    elif modelname == "lightgbm":
        # 'iterations'는 LightGBM 유효 alias가 아니라 무시된다 → n_estimators 사용
        # (get_search_space 쪽 주석 참고)
        params.update({
            'n_estimators': 10000,
            'early_stopping_rounds': 20,
            'verbosity': -1,
        })

    elif modelname in ["mlp", "resnet", "ftt", "embedmlp", "mlpplr", "saint", "t2gformer"]:
        # ⚠ n_epochs 는 100 고정이다. get_search_space() 가 large_datalist 여부와
        #   무관하게 100 을 쓰므로, 여기서만 50 으로 줄이면 large_datalist
        #   데이터셋에서 HPO(100 epoch)와 reproduce(50 epoch)가 어긋난다.
        params.update({
            'n_epochs': 100,
            'early_stopping_rounds': 20,
        })
        if modelname == "t2gformer":
            params.setdefault('token_bias', True)

    elif modelname in ["tabr", "modernnca"]:
        # ⚠ 이 두 모델은 params["model"] 중첩 구조와 최상위 리터럴을 함께
        #   복원해야 하므로 **전부 rearrange_params 가 처리한다**. 여기서
        #   early_stopping_rounds 를 건드리면 값이 어긋난다:
        #   get_search_space 기준으로 tabr 은 10, modernnca 는 20 이다.
        pass

    elif modelname == "tabm":
        # rearrange_params 가 처리한다
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

    # 3. TabR & ModernNCA 보정
    elif modelname in ("tabr", "modernnca"):
        # ⚠ 이 두 모델의 get_search_space() 는 중첩 dict 를 만든다:
        #       params["model"]["d_main"]  /  params["model"]["num_embeddings"][...]
        #   그런데 Optuna 의 study.best_params 는 **평탄한** dict 다.
        #   trial.suggest_int('d_main', ...) 로 등록된 "이름"만 남기 때문에
        #   reproduce 시점의 params 는
        #       {'d_main': 200, 'context_dropout': 0.3, 'd_embedding': 32, ...}
        #   이고 params["model"] 키가 아예 없다.
        #
        #   이전 버전은 params["model"] 을 새로 만들고 d_multiplier 등 고정값만
        #   넣었기 때문에, 탐색된 d_main / encoder_n_blocks / ... 가 최상위에
        #   남아 model 하위로 들어가지 못했다. 그 결과 tabr.py 의
        #   filtered_params 가 비어
        #       TypeError: TabR.__init__() missing 5 required keyword-only
        #                  arguments: 'd_main', 'encoder_n_blocks', ...
        #   가 발생했다. 여기서 평탄한 키를 다시 중첩 구조로 복원한다.
        #
        # ⚠ tabr 은 lr / weight_decay / early_stopping_rounds / lr_scheduler 까지
        #   전부 리터럴이라 best_params 에 하나도 없다. modernnca 는 lr /
        #   weight_decay / lr_scheduler 를 탐색하므로 남는다.
        _MODEL_KEYS = {
            "tabr":      ("d_main", "context_dropout", "encoder_n_blocks",
                          "predictor_n_blocks", "dropout0"),
            "modernnca": ("d_block", "dim", "dropout", "n_blocks"),
        }[modelname]
        _EMB_KEYS = ("d_embedding", "frequency_scale", "n_frequencies")

        mp = params.get("model")
        if not isinstance(mp, dict):
            mp = {}
        for k in _MODEL_KEYS:
            if k in params and k not in mp:
                mp[k] = params.pop(k)

        ne = mp.get("num_embeddings")
        if not isinstance(ne, dict):
            ne = {}
        for k in _EMB_KEYS:
            if k in params and k not in ne:
                ne[k] = params.pop(k)

        if modelname == "tabr":
            # suggest 된 d_embedding / frequency_scale / n_frequencies 는 위의
            # _EMB_KEYS 루프에서 중첩 구조로 복원된다. (과거 버전은 d_embedding
            # 을 64로 강제 덮어썼으나, 탐색 공간의 중복 키 버그가 수정되어
            # 더 이상 필요 없음. 다만 옛 로그에는 n_frequencies 가 없을 수
            # 있으므로 tabr.py 쪽 기본값 48이 폴백으로 사용된다.)
            mp.setdefault("d_multiplier", 2.0)
            mp.setdefault("mixer_normalization", "auto")
            mp.setdefault("dropout1", 0.0)
            mp.setdefault("normalization", "LayerNorm")
            mp.setdefault("activation", "ReLU")
            # 최상위 리터럴 (best_params 에 없음)
            params.setdefault("lr", 1e-4)
            params.setdefault("weight_decay", 1e-5)
            params.setdefault("lr_scheduler", True)
            params["early_stopping_rounds"] = 10   # ⚠ tabr 만 10, 나머지는 20
        else:  # modernnca
            if "n_blocks" not in mp:
                mp["n_blocks"] = 0     # large_set 에서는 탐색되지 않고 0 고정
            params.setdefault("early_stopping_rounds", 20)

        mp["num_embeddings"] = ne
        params["model"] = mp

    # 4. TabM 보정
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
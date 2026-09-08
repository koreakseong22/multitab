import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tokenize
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import torch
from scipy.special import expit, softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from libs.data import load_data, split_data
from libs.eval import calculate_metric, calculate_multi_auroc, is_study_todo
from libs.model import getmodel
from libs.runtime import check_implementation, implementation_id
from libs.search_space import get_search_space, rearrange_params
from libs.supervised import EarlyStopping
from libs.saint import TabAttention
from libs.tabr import TabR
import ensemble
import optimize
import reproduce


HPO_MODELS = ['randomforest', 'xgboost', 'catboost', 'lightgbm', 'mlp',
              'embedmlp', 'mlpplr', 'ftt', 'resnet', 't2gformer', 'saint',
              'modernnca', 'tabr']
DEEP_MODELS = HPO_MODELS[4:]


class SmallTrial:
    def __init__(self):
        self.params = {}

    def suggest_int(self, name, low, high, **kwargs):
        value = max(low, min(high, {'width': 16, 'd_embedding_num': 8}.get(name, low)))
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, **kwargs):
        value = max(low, min(high, {'learning_rate': 0.001, 'lr': 0.001}.get(name, low)))
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = {'lr_scheduler': False, 'activation': 'relu', 'normalization': 'batchnorm'}.get(name, choices[0])
        if value not in choices:
            value = choices[0]
        self.params[name] = value
        return value


class SyntheticDataset:
    def __init__(self, task='binclass', features='mixed', device='cpu'):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(96, 4)).astype('float32')
        self.X_num = list(range(4))
        self.X_cat = []
        self.X_cat_cardinality = []
        if features == 'mixed':
            X[:, 3] = np.arange(96) % 3
            self.X_num, self.X_cat, self.X_cat_cardinality = [0, 1, 2], [3], [3]
        elif features == 'categorical':
            X = rng.integers(0, 3, size=(96, 4)).astype('float32')
            self.X_num, self.X_cat, self.X_cat_cardinality = [], list(range(4)), [3] * 4
        if task == 'regression':
            y = (X[:, 0] * 0.4 + np.arange(96) / 96).astype('float32')
        elif task == 'multiclass':
            y = np.eye(3, dtype='float32')[np.arange(96) % 3]
        else:
            y = (np.arange(96) % 2).astype('float32')
        X, y = torch.tensor(X, device=device), torch.tensor(y, device=device)
        self.parts = [(X[:72], y[:72]), (X[72:84], y[72:84]), (X[84:], y[84:])]
        self.y_std = 2.0 if task == 'regression' else 1.0

    def _indv_dataset(self):
        return self.parts


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    def test_parameter_roundtrip(self):
        for name in HPO_MODELS:
            for data_id in (10, 422, 44159):
                with self.subTest(model=name, data_id=data_id):
                    trial = SmallTrial()
                    expected = get_search_space(trial, name, 4, data_id)
                    before = copy.deepcopy(trial.params)
                    actual = rearrange_params(name, data_id, trial.params, num_features=4)
                    self.assertEqual(actual, expected)
                    self.assertEqual(trial.params, before)
        with self.assertRaises(ValueError):
            rearrange_params('tabr', 10, {'d_main': 96}, num_features=4)

    def test_upstream_files_and_comments(self):
        root = Path(__file__).resolve().parent.parent
        manifest = json.loads((root / 'tests/upstream_manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            with self.subTest(file=name):
                content = (root / name).read_bytes().replace(b'\r\n', b'\n')
                if expected['unchanged']:
                    self.assertEqual(hashlib.sha256(content).hexdigest(), expected['sha256'])
                if name.endswith('.py'):
                    comments = [t.string for t in tokenize.generate_tokens(io.StringIO(content.decode()).readline)
                                if t.type == tokenize.COMMENT]
                    for added in expected['additional_comments']:
                        comments.remove(added)
                    digest = hashlib.sha256(json.dumps(comments, ensure_ascii=False).encode()).hexdigest()
                    self.assertEqual(digest, expected['comments_sha256'])

    def test_fold_and_preprocessing(self):
        X = np.arange(400, dtype=np.float32).reshape(100, 4)
        y = np.arange(100, dtype=np.float32)
        folds = list(KFold(10, shuffle=True, random_state=42).split(X))
        for seed in range(10):
            train, val, test, scale = split_data(X, y, 'regression', seed=seed, device='cpu')
            np.testing.assert_array_equal(test[0].numpy(), X[folds[seed][1]])
            np.testing.assert_array_equal(val[0].numpy(), X[folds[(seed + 1) % 10][1]])
            self.assertEqual([len(train[0]), len(val[0]), len(test[0])], [80, 10, 10])
            self.assertAlmostEqual(float(train[1].mean()), 0, places=5)
            self.assertGreater(scale, 0)
        with self.assertRaises(ValueError):
            split_data(X, y, 'regression', seed=10, device='cpu')

    def test_loader_removes_missing_targets_and_consecutive_bad_columns(self):
        class OpenMLDataset:
            name, default_target_attribute = 'synthetic', 'target'

            def get_data(self, target):
                X = pd.DataFrame({'numeric': [1., 2., 3., 4.],
                                  'bad1': ['a'] * 4, 'bad2': ['b'] * 4,
                                  'cat': pd.Categorical(['x', 'y', 'x', 'y'])})
                return X, pd.Series([0.25, np.nan, 0.75, 1.25]), [False, False, False, True], list(X.columns)

        with patch('libs.data.openml.datasets.get_dataset', return_value=OpenMLDataset()):
            X, y, cats, cardinality, nums = load_data(10)
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual((cats, cardinality, nums), ([1], [2], [0]))
        np.testing.assert_array_equal(y, [0.25, 0.75, 1.25])

    def test_probability_metrics(self):
        y = np.eye(3)[[0, 1, 2]]
        p = np.array([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7]])
        result = calculate_metric(y, np.arange(3), p, 'multiclass', 'test', prob=True)
        self.assertAlmostEqual(result['logloss_test'], log_loss(y, p))
        self.assertIsNone(calculate_multi_auroc(np.pad(y, ((0, 0), (0, 1))), np.pad(p, ((0, 0), (0, 1)))))
        binary = calculate_metric(np.array([0, 1]), np.array([[0], [1]]), np.array([[.8, .2], [.1, .9]]), 'binclass', 'test', prob=True)
        self.assertAlmostEqual(binary['logloss_test'], log_loss([0, 1], [.2, .9]))

    def test_binary_numeric_labels(self):
        X = np.ones((100, 2), dtype=np.float32)
        parts = split_data(X, np.arange(100) % 2 + 1, 'binclass', device='cpu')
        self.assertEqual(set(torch.cat([p[1] for p in parts[:3]]).tolist()), {0., 1.})

    def test_saint_tab_attention_uses_category_cardinalities(self):
        model = TabAttention(
            categories=[2, 3], num_continuous=2, dim=8, depth=1, heads=1
        )
        self.assertEqual(model.num_categories, 2)
        self.assertEqual(model.num_unique_categories, 5)
        torch.testing.assert_close(model.categories_offset, torch.tensor([1, 3]))

    def test_legacy_tabpfn_configuration_and_probability_contract(self):
        from libs.tabpfn import tabpfn

        class LegacyClassifier:
            def __init__(self, device, N_ensemble_configurations):
                self.configurations = N_ensemble_configurations

            def predict_proba(self, X):
                return np.array([[.8, .2], [.1, .9]])

        with patch('libs.tabpfn.TabPFNClassifier', LegacyClassifier):
            model = tabpfn('binclass')
            self.assertEqual(model.model.configurations, 32)
            X = torch.zeros(2, 1)
            np.testing.assert_allclose(expit(model.predict_proba(X, logit=True)), [.2, .9])
            np.testing.assert_allclose(model.predict_proba(X), [[.8, .2], [.1, .9]])

        class ModernClassifier:
            def __init__(self, device):
                raise AssertionError('A different model must not be constructed')

        with patch('libs.tabpfn.TabPFNClassifier', ModernClassifier):
            with self.assertRaisesRegex(RuntimeError, 'legacy TabPFN'):
                tabpfn('binclass')

    def test_optimization_and_reproduction_entry_points(self):
        class PerfectClassifier:
            def fit(self, *args):
                pass

            def predict(self, X):
                return np.arange(len(X)) % 2

            def predict_proba(self, X, logit=False):
                p = np.where(self.predict(X) == 1, .9, .1)
                return np.log(p / (1 - p)) if logit else p

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                dataset = SyntheticDataset()
                argv = ['optimize.py', '--gpu_id', '-1', '--modelname', 'mlp', '--openml_id', '10', '--seed', '1', '--savepath', directory]
                info = {'10': {'name': 'synthetic', 'tasktype': 'binclass'}}
                with patch.object(sys, 'argv', argv), patch.object(optimize, 'TabularDataset', return_value=dataset), patch.object(optimize, 'getmodel', return_value=PerfectClassifier()), patch.object(optimize.json, 'load', return_value=info), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    optimize.main()
                log = Path(directory) / 'optim_logs/seed=1/data=10..model=mlp.pkl'
                self.assertTrue(log.exists())
                study = optimize.joblib.load(log)
                self.assertEqual(study.best_value, 1.0)
                self.assertEqual(len(study.trials), 1)
                self.assertAlmostEqual(study.best_trial.user_attrs['logloss_val'], -np.log(.9))
                csv_log = log.with_suffix('.csv')
                csv_log.unlink()
                with patch.object(sys, 'argv', argv), patch.object(optimize, 'TabularDataset', side_effect=AssertionError('completed study must not retrain')), patch.object(optimize.json, 'load', return_value=info), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    optimize.main()
                self.assertTrue(csv_log.exists())
                argv = ['reproduce.py', '--gpu_id', '-1', '--openml_id', '10', '--seed', '1', '--savepath', directory]
                with patch.object(sys, 'argv', argv), patch.object(reproduce, 'TabularDataset', return_value=dataset), patch.object(reproduce, 'getmodel', return_value=PerfectClassifier()), patch.object(reproduce.json, 'load', return_value=info), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    reproduce.main()
                output = Path(directory) / 'reproduce_logs/seed=1/data=10/model=mlp..init_hps=False..deep=0..hyper=0.npy'
                prediction = np.load(output, allow_pickle=True).item()
                self.assertAlmostEqual(prediction['Performance']['logloss_test'], -np.log(.9))
                self.assertEqual(prediction['implementation_id'], implementation_id())
                self.assertTrue((output.parent / 'model=mlp..init_hps=False..deep=4..hyper=0.npy').exists())
            finally:
                os.chdir(previous)

    def test_empty_study_and_early_stopping(self):
        self.assertTrue(is_study_todo(optuna.create_study(direction='maximize'), 'binclass'))
        stopper = EarlyStopping(patience=2)
        stopper.on_epoch_end(1, {'val_loss': 1.0})
        self.assertEqual(stopper.patience_counter, 0)
        stopper.on_epoch_end(2, {'val_loss': 1.0})
        self.assertFalse(stopper.should_stop)
        stopper.on_epoch_end(3, {'val_loss': 1.0})
        self.assertTrue(stopper.should_stop)

    def test_provenance(self):
        signature = implementation_id()
        check_implementation({'implementation_id': signature}, signature)
        with self.assertRaises(ValueError):
            check_implementation({}, signature)
        check_implementation({}, signature, allow_legacy=True)
        with self.assertRaises(ValueError):
            check_implementation({'implementation_id': 'old'}, signature, allow_legacy=True)

    def _train_model(self, name, task, features='mixed', device='cpu'):
        dataset = SyntheticDataset(task, features, device)
        train, val, test = dataset.parts
        params = {} if name in ['lr', 'knn', 'dt'] else get_search_space(SmallTrial(), name, 4, 10)
        if 'n_epochs' in params:
            params['n_epochs'] = 1
        if 'n_estimators' in params:
            params['n_estimators'] = 4
        if 'iterations' in params:
            params['iterations'] = 4
        if name == 'catboost':
            params['allow_writing_files'] = False
            params['thread_count'] = 1
        if name in ['lightgbm', 'xgboost', 'randomforest']:
            params['n_jobs'] = 1
        model = getmodel(name, params, task, dataset, 10, 4, 3 if task == 'multiclass' else 1, torch.device(device))
        if name in ['tabr', 'modernnca']:
            model.max_epoch = 1
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(*train, *val)
            prediction = np.asarray(model.predict(test[0]))
            probabilities = model.predict_proba(test[0]) if task != 'regression' else None
            logits = model.predict_proba(test[0], logit=True) if task != 'regression' else None
        self.assertTrue(np.isfinite(prediction).all())
        metrics = calculate_metric(test[1], prediction, probabilities, task, 'test', prob=True)
        self.assertTrue(np.isfinite(metrics['rmse_test' if task == 'regression' else 'acc_test']))
        if task != 'regression':
            probabilities = np.asarray(probabilities)
            self.assertTrue(np.isfinite(probabilities).all())
            if task == 'multiclass':
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                np.testing.assert_allclose(softmax(logits, axis=1), probabilities, atol=1e-5)
            else:
                positive = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] == 2 else probabilities.reshape(-1)
                actual = np.asarray(logits).reshape(-1) if name == 'modernnca' else expit(np.asarray(logits).reshape(-1))
                np.testing.assert_allclose(actual, positive, atol=1e-5)

    def test_model_training_and_predictions(self):
        for task in ['binclass', 'multiclass', 'regression']:
            for name in ['lr', 'knn', 'dt'] + HPO_MODELS:
                with self.subTest(task=task, model=name):
                    self._train_model(name, task)

    def test_deep_feature_extremes(self):
        for features in ['numeric', 'categorical']:
            for name in DEEP_MODELS:
                with self.subTest(features=features, model=name):
                    self._train_model(name, 'binclass', features)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_model_paths(self):
        for name in DEEP_MODELS:
            with self.subTest(model=name):
                self._train_model(name, 'binclass', device='cuda:0')

    def test_tabr_index_alignment_and_self_exclusion(self):
        params = get_search_space(SmallTrial(), 'tabr', 1, 10)['model']
        params['d_main'] = 4
        model = TabR(n_num_features=1, n_cat_features=0, n_classes=1, **params)
        model._encode = lambda xn, xc: (torch.zeros(len(xn), 4), torch.cat([xn, torch.zeros(len(xn), 3)], dim=1))
        candidates = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
        model.cached_candidate_k = model._encode(candidates, None)[1]
        model.cached_candidate_y = torch.arange(5, dtype=torch.float32)
        model.update_index()
        seen = []
        handle = model.label_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        model(x_num=candidates[[3, 1]], x_cat=None, y=torch.tensor([3., 1.]),
              candidate_x_num=candidates, candidate_x_cat=None, candidate_y=model.cached_candidate_y,
              context_size=96, is_train=True, query_indices=torch.tensor([3, 1]))
        handle.remove()
        self.assertEqual(seen[0].shape, (2, 4, 1))
        self.assertNotIn(3., seen[0][0].flatten().tolist())
        self.assertNotIn(1., seen[0][1].flatten().tolist())

    def test_ensemble_files_metrics_and_repeated_runs(self):
        for task in ['binclass', 'multiclass', 'regression']:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                dataset = SyntheticDataset(task)
                target = dataset.parts[2][1].numpy()
                logs = Path(directory) / 'reproduce_logs/seed=1/data=10'
                logs.mkdir(parents=True)
                for option in ensemble.opts['all']:
                    if task == 'regression':
                        probability = None
                        prediction = target + 0.25
                    elif task == 'multiclass':
                        probability = target * 2
                        prediction = target.argmax(axis=1)
                    else:
                        probability = (target * 2 - 1) * 2
                        prediction = target
                    np.save(logs / f'model=mlp..init_hps=False..{option}.npy',
                            {'Probability': probability, 'Prediction': prediction, 'implementation_id': implementation_id()})
                with contextlib.redirect_stdout(io.StringIO()):
                    for _ in range(2):
                        for kind in ['deep', 'hyper', 'all']:
                            ensemble.get_results('out.csv', dataset, task, 1, '10', 'mlp', kind, directory)
                result = pd.read_csv(Path(directory) / 'out.csv')
                self.assertEqual(len(result), 9)
                self.assertTrue((Path(directory) / 'ensemble_logs/seed=1/data=10/model=mlp..init_hps=False..deep=5..hyper=5.npy').exists())
                self.assertTrue(np.allclose(result.acc_rmse, 0.5 if task == 'regression' else 1))

    def test_ensemble_does_not_duplicate_missing_members(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / 'reproduce_logs/seed=1/data=10'
            logs.mkdir(parents=True)
            dataset = SyntheticDataset()
            np.save(logs / 'model=mlp..init_hps=False..deep=0..hyper=0.npy',
                    {'Probability': np.zeros(12), 'implementation_id': implementation_id()})
            with contextlib.redirect_stdout(io.StringIO()):
                ensemble.get_results('out.csv', dataset, 'binclass', 1, '10', 'mlp', basepath=directory)
            self.assertFalse((Path(directory) / 'out.csv').exists())


if __name__ == '__main__':
    unittest.main()

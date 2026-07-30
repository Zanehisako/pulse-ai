# Cell 1 — Imports, configuration, optional SOTA packages, GPU setup

import os, gc, json, math, zipfile, random, warnings, sys, subprocess, importlib.util, shutil
from pathlib import Path
from tqdm.auto import tqdm
from importlib import metadata as importlib_metadata
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

SEED = 20260511
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("models")
DATA_DIR = Path("generated_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "validation_protocol_version": "real_ops_sota_v2",
    "smoke_mode": os.getenv("PIOS_SMOKE_MODE", "0") == "1",
    "inventory_days": 455,
    "inventory_hospitals": 42,
    "n_donors": 90000,
    "donor_snapshots": 8,
    "batch_size": 4096,
    "predict_batch_size": 16384,
    "gradient_accumulation_steps": 1,
    "epochs": 30,
    "patience": 6,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 3,
    "hidden_dim": 384,
    "ffn_multiplier": 4/3,
    "n_transformer_layers": 3,
    "n_heads": 8,
    "dropout": 0.15,
    "n_plr_bins": 64,
    "max_rows_neural": 650000,
    "neural_loader_workers": 4,
    "compile_model": True,
    "hpo_trials": int(os.getenv("PIOS_HPO_TRIALS", "12")),
    "hpo_timeout_seconds": int(os.getenv("PIOS_HPO_TIMEOUT_SECONDS", "900")),
    "grid_search_enabled": os.getenv("PIOS_GRID_SEARCH", "1") == "1",
    "grid_search_tabular": {
        "num_leaves": [63, 127, 255],
        "learning_rate": [0.01, 0.03, 0.06],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.80, 0.95],
        "colsample_bytree": [0.80, 0.95],
        "reg_alpha": [0.001, 0.02, 0.5],
        "reg_lambda": [0.1, 1.2, 3.0],
    },
    "grid_search_neural": {
        "lr": [1e-4, 3e-4, 5e-4],
        "dropout": [0.10, 0.15, 0.25],
        "hidden_dim": [256, 384],
        "n_transformer_layers": [2, 3, 4],
    },
    "grid_search_max_tabular_combos": int(os.getenv("PIOS_GRID_MAX_TABULAR", "12")),
    "grid_search_max_neural_combos": int(os.getenv("PIOS_GRID_MAX_NEURAL", "3")),
    "enable_optional_sota_baselines": os.getenv("PIOS_ENABLE_OPTIONAL_SOTA", "1") == "1",
    "allow_optional_installs": os.getenv("PIOS_ALLOW_OPTIONAL_INSTALLS", "1") == "1",
    "optional_install_packages": [p.strip() for p in os.getenv("PIOS_OPTIONAL_INSTALL_PACKAGES", "tabpfn,autogluon,tabm").split(",") if p.strip()],
    "optional_install_quiet": True,
    "max_rows_tabpfn": 50000,
    "max_rows_autogluon": 250000,
    "autogluon_time_limit_seconds": int(os.getenv("PIOS_AUTOGLUON_TIME_LIMIT_SECONDS", "600")),
    "enable_experimental_tabm_adapter": os.getenv("PIOS_ENABLE_TABM_ADAPTER", "0") == "1",
    "enable_experimental_tabr_adapter": os.getenv("PIOS_ENABLE_TABR_ADAPTER", "0") == "1",
    "allow_proxy_derived_donor_features": os.getenv("PIOS_ALLOW_PROXY_DONOR_FEATURES", "0") == "1",
    "real_data_paths": {
        "inventory_features": os.getenv("PIOS_REAL_INVENTORY_FEATURES", ""),
        "inventory_labels": os.getenv("PIOS_REAL_INVENTORY_LABELS", ""),
        "donor_features": os.getenv("PIOS_REAL_DONOR_FEATURES", ""),
        "donor_labels": os.getenv("PIOS_REAL_DONOR_LABELS", ""),
    },
    "min_auc_gate": 0.80,
    "min_average_precision_gate": 0.70,
    "max_brier_gate": 0.18,
    "max_ece_gate": 0.08,
    "max_wape_gate": 0.25,
    "max_coverage_error": 0.08,
    "max_worst_group_wape_gate": 0.35,
    "max_worst_group_brier_gate": 0.22,
    "min_rare_type_recall_gate": 0.70,
}
if CONFIG["smoke_mode"]:
    CONFIG.update({
        "inventory_days": 90,
        "inventory_hospitals": 12,
        "n_donors": 8000,
        "donor_snapshots": 3,
        "epochs": 3,
        "patience": 2,
        "hpo_trials": 1,
        "hpo_timeout_seconds": 90,
        "grid_search_max_tabular_combos": 3,
        "grid_search_max_neural_combos": 2,
        "autogluon_time_limit_seconds": 60,
        "max_rows_neural": 80000,
        "max_rows_tabpfn": 8000,
        "max_rows_autogluon": 30000,
    })

OPTIONAL_DEPENDENCIES = {
    "tabpfn": {"import_name": "tabpfn", "pip_names": ["tabpfn"], "enabled": CONFIG["enable_optional_sota_baselines"]},
    "autogluon": {"import_name": "autogluon.tabular", "pip_names": ["autogluon.tabular", "autogluon"], "enabled": CONFIG["enable_optional_sota_baselines"]},
    "optuna": {"import_name": "optuna", "pip_names": ["optuna"], "enabled": True},
    "tabm": {"import_name": "tabm", "pip_names": ["tabm"], "enabled": CONFIG["enable_optional_sota_baselines"]},
}

def optional_version(import_name):
    root = import_name.split(".")[0]
    try:
        return importlib_metadata.version(root)
    except Exception:
        return None

def ensure_optional_dependency(key):
    spec = OPTIONAL_DEPENDENCIES[key]
    if not spec["enabled"]:
        return {"available": False, "reason": "disabled", "version": None}
    try:
        import_available = importlib.util.find_spec(spec["import_name"]) is not None
    except Exception:
        import_available = False
    if not import_available and CONFIG["allow_optional_installs"] and key in CONFIG["optional_install_packages"]:
        install_errors = []
        for pip_name in spec["pip_names"]:
            cmd = [sys.executable, "-m", "pip", "install", "--prefer-binary", pip_name]
            if CONFIG["optional_install_quiet"]:
                cmd.append("-q")
            try:
                subprocess.check_call(cmd)
                break
            except Exception as exc:
                install_errors.append(f"{pip_name}:{type(exc).__name__}")
        if install_errors:
            try:
                import_available = importlib.util.find_spec(spec["import_name"]) is not None
            except Exception:
                import_available = False
            if not import_available:
                return {"available": False, "reason": "install_failed:" + ",".join(install_errors), "version": None}
    elif not import_available and CONFIG["allow_optional_installs"] and key not in CONFIG["optional_install_packages"]:
        return {"available": False, "reason": "install_skipped_by_config", "version": None}
    try:
        available = importlib.util.find_spec(spec["import_name"]) is not None
    except Exception:
        available = False
    return {"available": available, "reason": None if available else "not_installed", "version": optional_version(spec["import_name"])}

OPTIONAL_PACKAGE_STATUS = {k: ensure_optional_dependency(k) for k in OPTIONAL_DEPENDENCIES}

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    HAS_TABPFN = OPTIONAL_PACKAGE_STATUS["tabpfn"]["available"]
except Exception as exc:
    HAS_TABPFN = False
    OPTIONAL_PACKAGE_STATUS["tabpfn"]["available"] = False
    OPTIONAL_PACKAGE_STATUS["tabpfn"]["reason"] = f"import_failed:{type(exc).__name__}"

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = OPTIONAL_PACKAGE_STATUS["autogluon"]["available"]
except Exception as exc:
    HAS_AUTOGLUON = False
    OPTIONAL_PACKAGE_STATUS["autogluon"]["available"] = False
    OPTIONAL_PACKAGE_STATUS["autogluon"]["reason"] = f"import_failed:{type(exc).__name__}"

try:
    import optuna
    HAS_OPTUNA = OPTIONAL_PACKAGE_STATUS["optuna"]["available"]
    if HAS_OPTUNA:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception as exc:
    HAS_OPTUNA = False
    OPTIONAL_PACKAGE_STATUS["optuna"]["available"] = False
    OPTIONAL_PACKAGE_STATUS["optuna"]["reason"] = f"import_failed:{type(exc).__name__}"

try:
    import tabm
    HAS_TABM = OPTIONAL_PACKAGE_STATUS["tabm"]["available"]
except Exception as exc:
    HAS_TABM = False
    OPTIONAL_PACKAGE_STATUS["tabm"]["available"] = False
    OPTIONAL_PACKAGE_STATUS["tabm"]["reason"] = f"import_failed:{type(exc).__name__}"

HAS_TABR = False
OPTIONAL_PACKAGE_STATUS["tabr"] = {
    "available": False,
    "reason": "research_reference_no_stable_notebook_adapter",
    "version": None,
}

def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print("CUDA:", torch.cuda.is_available(), "Device:", DEVICE)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
print("LightGBM:", HAS_LGB, "XGBoost:", HAS_XGB, "CatBoost:", HAS_CAT)
print("Optional SOTA packages:", json.dumps(OPTIONAL_PACKAGE_STATUS, indent=2))


# Cell 3 — Metrics, splits, encoders, utilities

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y, float), np.asarray(p, float))))

def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(np.abs(y-p)) / max(np.sum(np.abs(y)), 1e-9))

def regression_metrics(y, p, baseline=None):
    out = {"rmse": rmse(y, p), "mae": float(mean_absolute_error(y, p)),
           "r2": float(r2_score(y, p)), "wape": wape(y, p)}
    if baseline is not None:
        out["baseline_wape"] = wape(y, baseline)
        out["wape_improvement_pct"] = float(100*(out["baseline_wape"]-out["wape"])/max(out["baseline_wape"], 1e-9))
    return out

def expected_calibration_error(y, prob, n_bins=10):
    y = np.asarray(y).astype(int)
    prob = np.clip(np.asarray(prob, float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(prob[mask]) - np.mean(y[mask])))
    return float(ece)

def classification_metrics(y, prob, threshold=.5):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "brier": float(brier_score_loss(y, np.clip(prob, 0, 1))),
           "ece": expected_calibration_error(y, prob),
           "positive_rate": float(np.mean(y)),
           "threshold": float(threshold)}
    out["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else .5
    out["average_precision"] = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else 0.0
    return out

def best_f1_threshold(y, prob):
    best_t, best = .5, -1
    for t in np.linspace(.02, .98, 49):
        f = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f > best:
            best_t, best = float(t), float(f)
    return best_t

def pinball(y, p, q):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = y - p
    return float(np.mean(np.maximum(q*d, (q-1)*d)))

def quantile_metrics(y, q10, q50, q90, baseline=None):
    out = regression_metrics(y, q50, baseline)
    out.update({
        "q10_pinball": pinball(y, q10, .1),
        "q50_pinball": pinball(y, q50, .5),
        "q90_pinball": pinball(y, q90, .9),
        "q10_q90_coverage": float(np.mean((np.asarray(y) >= np.asarray(q10)) & (np.asarray(y) <= np.asarray(q90)))),
        "coverage_error_vs_80pct": float(abs(np.mean((np.asarray(y) >= np.asarray(q10)) & (np.asarray(y) <= np.asarray(q90))) - .80)),
        "q10_q90_avg_width": float(np.mean(np.asarray(q90) - np.asarray(q10))),
    })
    return out

def temporal_split(df, time_col="event_timestamp", train=.70, valid=.15):
    x = df.sort_values(time_col).reset_index(drop=True)
    n = len(x)
    a = int(n*train)
    b = int(n*(train+valid))
    return x.iloc[:a].copy(), x.iloc[a:b].copy(), x.iloc[b:].copy()

def donor_group_holdout_split(df, entity_col="donor_id", time_col="event_timestamp", test_frac=.15, valid_frac=.15):
    ent = np.array(sorted(df[entity_col].astype(str).unique()))
    rng = np.random.default_rng(SEED + 101)
    rng.shuffle(ent)
    n_test = max(1, int(len(ent)*test_frac))
    n_valid = max(1, int(len(ent)*valid_frac))
    test_e, valid_e = set(ent[:n_test]), set(ent[n_test:n_test+n_valid])
    train_e = set(ent[n_test+n_valid:])
    return (df[df[entity_col].astype(str).isin(train_e)].sort_values(time_col).copy(),
            df[df[entity_col].astype(str).isin(valid_e)].sort_values(time_col).copy(),
            df[df[entity_col].astype(str).isin(test_e)].sort_values(time_col).copy())

def unseen_hospital_split(df, entity_col="hospital_id", time_col="event_timestamp", test_frac=.20, valid_frac=.15):
    ent = np.array(sorted(df[entity_col].astype(str).unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ent)
    n_test = max(1, int(len(ent)*test_frac))
    n_valid = max(1, int(len(ent)*valid_frac))
    test_e, valid_e = set(ent[:n_test]), set(ent[n_test:n_test+n_valid])
    train_e = set(ent[n_test+n_valid:])
    return (df[df[entity_col].astype(str).isin(train_e)].sort_values(time_col).copy(),
            df[df[entity_col].astype(str).isin(valid_e)].sort_values(time_col).copy(),
            df[df[entity_col].astype(str).isin(test_e)].sort_values(time_col).copy())

def product_key(comp, bt):
    return f"{comp} | {bt}"

def group_metrics_quantile(test, y, q10, q50, q90, baseline):
    d = test[["component_type", "blood_type"]].copy()
    d["y"] = np.asarray(y)
    d["q10"] = q10
    d["q50"] = q50
    d["q90"] = q90
    d["baseline"] = baseline
    out = {}
    for (c, b), g in d.groupby(["component_type", "blood_type"]):
        key = product_key(c, b)
        out[key] = quantile_metrics(g.y, g.q10, g.q50, g.q90, g.baseline)
        out[key].update({"component_type": c, "blood_type": b, "rows": int(len(g))})
    return out

def group_metrics_reg(test, y, pred, baseline):
    d = test[["component_type", "blood_type"]].copy()
    d["y"] = np.asarray(y)
    d["p"] = pred
    d["baseline"] = baseline
    out = {}
    for (c, b), g in d.groupby(["component_type", "blood_type"]):
        key = product_key(c, b)
        out[key] = regression_metrics(g.y, g.p, g.baseline)
        out[key].update({"component_type": c, "blood_type": b, "rows": int(len(g))})
    return out

def group_metrics_cls(test, y, prob, threshold=.5):
    d = test[["component_type", "blood_type"]].copy()
    d["y"] = np.asarray(y).astype(int)
    d["p"] = prob
    out = {}
    for (c, b), g in d.groupby(["component_type", "blood_type"]):
        key = product_key(c, b)
        out[key] = classification_metrics(g.y, g.p, threshold)
        out[key].update({"component_type": c, "blood_type": b, "rows": int(len(g))})
    return out

def rolling_baseline(train, test, target, group_cols=["hospital_id", "blood_type", "component_type"], window=7):
    hist = train.sort_values("event_timestamp")
    roll = hist.groupby(group_cols)[target].apply(lambda s: s.rolling(window, min_periods=1).mean().iloc[-1])
    roll.name = "_baseline"
    default = float(train[target].median())
    merged = test[group_cols].merge(roll.reset_index(), on=group_cols, how="left")
    return merged["_baseline"].fillna(default).to_numpy()

def leakage_audit(feature_cols, blocked, name):
    bad = []
    for c in feature_cols:
        lo = c.lower()
        for pat in blocked:
            p = pat.lower()
            if p.startswith("*") and p.endswith("*") and p.strip("*") in lo:
                bad.append((c, pat))
            elif p == lo:
                bad.append((c, pat))
    if bad:
        raise ValueError(f"Leakage audit failed for {name}: {bad[:20]}")
    print("Leakage audit passed:", name, len(feature_cols), "features")

DONOR_PROXY_DERIVED_FEATURES = [
    "readiness_score", "response_readiness_index", "donor_momentum", "operational_outreach_priority",
]

def apply_donor_proxy_policy(feature_cols, name):
    found = [c for c in feature_cols if c in DONOR_PROXY_DERIVED_FEATURES]
    if not found:
        return feature_cols
    if CONFIG["allow_proxy_derived_donor_features"]:
        print("Proxy-derived donor feature audit allowed:", {"check": name, "proxy_derived_features": found})
        return feature_cols
    filtered = [c for c in feature_cols if c not in DONOR_PROXY_DERIVED_FEATURES]
    print("Proxy-derived donor features removed for leakage control:", {"check": name, "proxy_derived_features": found})
    return filtered

def worst_group_metric(group_metrics, metric, higher_is_better=False):
    vals = [v.get(metric) for v in group_metrics.values() if metric in v and v.get("rows", 0) > 0]
    if not vals:
        return None
    return float(min(vals) if higher_is_better else max(vals))

def evaluate_regression_gate(metrics, group_metrics=None):
    return {
        "wape_pass": metrics.get("wape", float("inf")) <= CONFIG["max_wape_gate"],
        "coverage_pass": metrics.get("coverage_error_vs_80pct", 0.0) <= CONFIG["max_coverage_error"],
        "worst_group_wape": worst_group_metric(group_metrics or {}, "wape", False),
        "worst_group_wape_pass": (worst_group_metric(group_metrics or {}, "wape", False) or 0.0) <= CONFIG["max_worst_group_wape_gate"],
    }

def evaluate_classification_gate(metrics, group_metrics=None):
    return {
        "auc_pass": metrics.get("auc", 0.0) >= CONFIG["min_auc_gate"],
        "average_precision_pass": metrics.get("average_precision", 0.0) >= CONFIG["min_average_precision_gate"],
        "brier_pass": metrics.get("brier", float("inf")) <= CONFIG["max_brier_gate"],
        "ece_pass": metrics.get("ece", float("inf")) <= CONFIG["max_ece_gate"],
        "worst_group_brier": worst_group_metric(group_metrics or {}, "brier", False),
        "worst_group_brier_pass": (worst_group_metric(group_metrics or {}, "brier", False) or 0.0) <= CONFIG["max_worst_group_brier_gate"],
    }

def save_bundle(name, bundle, metrics):
    joblib.dump(bundle, OUTPUT_DIR / f"{name}.joblib")
    save_json(metrics, OUTPUT_DIR / f"{name}_metrics.json")
    print("Saved", name)


# Cell 4 — Tabular model comparison helpers

def encode_ohe(train, valid, test, feature_cols):
    cols = list(dict.fromkeys(feature_cols))

    def prep(x):
        x = x[cols].copy()
        for c in x.columns:
            if x[c].dtype == "object" or str(x[c].dtype).startswith("category"):
                x[c] = x[c].astype(str).fillna("__MISSING__")
            elif pd.api.types.is_datetime64_any_dtype(x[c]):
                x[c] = pd.to_datetime(x[c], utc=True).astype("int64")/1e9
            else:
                x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype("float32")
        return pd.get_dummies(x)

    tr, va, te = prep(train), prep(valid), prep(test)
    cols2 = list(tr.columns)
    va = va.reindex(columns=cols2, fill_value=0)
    te = te.reindex(columns=cols2, fill_value=0)
    return tr.astype("float32"), va.astype("float32"), te.astype("float32"), cols2

def sample_rows(x, y, max_rows):
    if len(x) <= max_rows:
        return x, y
    rng = np.random.default_rng(SEED + len(x))
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x.iloc[idx], np.asarray(y)[idx]

def _grid_sample_combos(grid, max_combos, seed=SEED):
    keys = list(grid.keys())
    all_combos = list(iterproduct(*[grid[k] for k in keys]))
    if len(all_combos) <= max_combos:
        return [dict(zip(keys, c)) for c in all_combos]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_combos), max_combos, replace=False)
    return [dict(zip(keys, all_combos[i])) for i in idx]

def tune_lgbm_regressor_params(train_x, y_train, valid_x, y_valid, objective="regression", alpha=None):
    params = dict(n_estimators=2000, learning_rate=.03, num_leaves=127, subsample=.90, colsample_bytree=.90,
                  reg_alpha=.02, reg_lambda=1.2, random_state=SEED, n_jobs=-1, verbose=-1)
    params.update(objective="quantile" if objective == "quantile" else "regression")
    if objective == "quantile":
        params["alpha"] = alpha
    if not (CONFIG.get("grid_search_enabled", False) and HAS_LGB):
        return params, {"enabled": False, "reason": "grid_search_disabled_or_lgb_unavailable"}
    grid = CONFIG["grid_search_tabular"]
    combos = _grid_sample_combos(grid, CONFIG["grid_search_max_tabular_combos"])
    print(f"    [Grid Search] {len(combos)} param combos for regressor")
    best_score, best_combo = float("inf"), None
    for combo in combos:
        trial_params = dict(params)
        trial_params.update(combo)
        model = lgb.LGBMRegressor(**trial_params)
        model.fit(train_x, y_train, eval_set=[(valid_x, y_valid)], callbacks=[lgb.early_stopping(30, verbose=False)])
        score = rmse(y_valid, model.predict(valid_x))
        if score < best_score:
            best_score, best_combo = score, combo
    params.update(best_combo)
    return params, {"enabled": True, "method": "grid_search", "best_value": float(best_score), "combos_tested": len(combos)}

def tune_lgbm_classifier_params(train_x, y_train, valid_x, y_valid, scale_pos_weight):
    params = dict(n_estimators=2000, learning_rate=.03, num_leaves=127, subsample=.9, colsample_bytree=.9,
                  reg_alpha=.02, reg_lambda=1.2, scale_pos_weight=scale_pos_weight,
                  random_state=SEED, n_jobs=-1, verbose=-1)
    if not (CONFIG.get("grid_search_enabled", False) and HAS_LGB):
        return params, {"enabled": False, "reason": "grid_search_disabled_or_lgb_unavailable"}
    grid = CONFIG["grid_search_tabular"]
    combos = _grid_sample_combos(grid, CONFIG["grid_search_max_tabular_combos"])
    print(f"    [Grid Search] {len(combos)} param combos for classifier")
    best_score, best_combo = -1.0, None
    for combo in combos:
        trial_params = dict(params)
        trial_params.update(combo)
        model = lgb.LGBMClassifier(**trial_params)
        model.fit(train_x, y_train, eval_set=[(valid_x, y_valid)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        p = model.predict_proba(valid_x)[:, 1]
        score = roc_auc_score(y_valid, p) if len(np.unique(y_valid)) > 1 else .5
        if score > best_score:
            best_score, best_combo = score, combo
    params.update(best_combo)
    return params, {"enabled": True, "method": "grid_search", "best_value": float(best_score), "combos_tested": len(combos)}

def fit_sota_regression_baselines(train, valid, test, feature_cols, target, task_name):
    out = {"valid": {}, "test": {}, "models": {}, "availability": json.loads(json.dumps(OPTIONAL_PACKAGE_STATUS))}
    if not CONFIG["enable_optional_sota_baselines"]:
        out["availability"]["all"] = {"available": False, "reason": "disabled", "version": None}
        return out
    tx, vx, tex, cols = encode_ohe(train, valid, test, feature_cols)
    y_train, y_valid, y_test = train[target].to_numpy(), valid[target].to_numpy(), test[target].to_numpy()
    if HAS_TABPFN and len(tx) <= CONFIG["max_rows_tabpfn"]:
        try:
            model = TabPFNRegressor(device=str(DEVICE))
        except TypeError:
            model = TabPFNRegressor()
        try:
            sx, sy = sample_rows(tx, y_train, CONFIG["max_rows_tabpfn"])
            model.fit(sx, sy)
            vpred = np.asarray(model.predict(vx)).reshape(-1)
            tpred = np.asarray(model.predict(tex)).reshape(-1)
            out["valid"]["tabpfn"] = regression_metrics(y_valid, vpred)
            out["test"]["tabpfn"] = regression_metrics(y_test, tpred)
            out["models"]["tabpfn"] = model
        except Exception as exc:
            out["availability"]["tabpfn"]["available"] = False
            out["availability"]["tabpfn"]["reason"] = f"fit_failed:{type(exc).__name__}"
    elif HAS_TABPFN:
        out["availability"]["tabpfn"]["available"] = False
        out["availability"]["tabpfn"]["reason"] = "row_limit_exceeded"
    if HAS_AUTOGLUON and len(train) <= CONFIG["max_rows_autogluon"]:
        try:
            ag_train = train[feature_cols + [target]].copy()
            ag_valid = valid[feature_cols + [target]].copy()
            predictor = TabularPredictor(label=target, problem_type="regression",
                                         path=str(OUTPUT_DIR / f"_autogluon_{task_name}"), verbosity=0)
            predictor.fit(ag_train, tuning_data=ag_valid, time_limit=CONFIG["autogluon_time_limit_seconds"],
                          presets="best_quality", use_bag_holdout=True)
            vpred = predictor.predict(valid[feature_cols])
            tpred = predictor.predict(test[feature_cols])
            out["valid"]["autogluon"] = regression_metrics(y_valid, vpred)
            out["test"]["autogluon"] = regression_metrics(y_test, tpred)
            out["models"]["autogluon"] = predictor
        except Exception as exc:
            out["availability"]["autogluon"]["available"] = False
            out["availability"]["autogluon"]["reason"] = f"fit_failed:{type(exc).__name__}"
    elif HAS_AUTOGLUON:
        out["availability"]["autogluon"]["available"] = False
        out["availability"]["autogluon"]["reason"] = "row_limit_exceeded"
    if HAS_TABM and not CONFIG["enable_experimental_tabm_adapter"]:
        out["availability"]["tabm"]["reason"] = "installed_but_adapter_disabled"
    if HAS_TABR and not CONFIG["enable_experimental_tabr_adapter"]:
        out["availability"]["tabr"]["reason"] = "adapter_disabled"
    return out

def fit_sota_classification_baselines(train, valid, test, feature_cols, target, task_name):
    out = {"valid": {}, "test": {}, "models": {}, "thresholds": {}, "availability": json.loads(json.dumps(OPTIONAL_PACKAGE_STATUS))}
    if not CONFIG["enable_optional_sota_baselines"]:
        out["availability"]["all"] = {"available": False, "reason": "disabled", "version": None}
        return out
    tx, vx, tex, cols = encode_ohe(train, valid, test, feature_cols)
    y_train = train[target].astype(int).to_numpy()
    y_valid = valid[target].astype(int).to_numpy()
    y_test = test[target].astype(int).to_numpy()
    if HAS_TABPFN and len(tx) <= CONFIG["max_rows_tabpfn"]:
        try:
            model = TabPFNClassifier(device=str(DEVICE))
        except TypeError:
            model = TabPFNClassifier()
        try:
            sx, sy = sample_rows(tx, y_train, CONFIG["max_rows_tabpfn"])
            model.fit(sx, sy)
            vprob = np.asarray(model.predict_proba(vx))[:, 1]
            tprob = np.asarray(model.predict_proba(tex))[:, 1]
            thr = best_f1_threshold(y_valid, vprob)
            out["thresholds"]["tabpfn"] = thr
            out["valid"]["tabpfn"] = classification_metrics(y_valid, vprob, thr)
            out["test"]["tabpfn"] = classification_metrics(y_test, tprob, thr)
            out["models"]["tabpfn"] = model
        except Exception as exc:
            out["availability"]["tabpfn"]["available"] = False
            out["availability"]["tabpfn"]["reason"] = f"fit_failed:{type(exc).__name__}"
    elif HAS_TABPFN:
        out["availability"]["tabpfn"]["available"] = False
        out["availability"]["tabpfn"]["reason"] = "row_limit_exceeded"
    if HAS_AUTOGLUON and len(train) <= CONFIG["max_rows_autogluon"]:
        try:
            ag_train = train[feature_cols + [target]].copy()
            ag_valid = valid[feature_cols + [target]].copy()
            predictor = TabularPredictor(label=target, problem_type="binary",
                                         path=str(OUTPUT_DIR / f"_autogluon_{task_name}"), verbosity=0)
            predictor.fit(ag_train, tuning_data=ag_valid, time_limit=CONFIG["autogluon_time_limit_seconds"],
                          presets="best_quality", use_bag_holdout=True)
            vprob = predictor.predict_proba(valid[feature_cols]).iloc[:, 1].to_numpy()
            tprob = predictor.predict_proba(test[feature_cols]).iloc[:, 1].to_numpy()
            thr = best_f1_threshold(y_valid, vprob)
            out["thresholds"]["autogluon"] = thr
            out["valid"]["autogluon"] = classification_metrics(y_valid, vprob, thr)
            out["test"]["autogluon"] = classification_metrics(y_test, tprob, thr)
            out["models"]["autogluon"] = predictor
        except Exception as exc:
            out["availability"]["autogluon"]["available"] = False
            out["availability"]["autogluon"]["reason"] = f"fit_failed:{type(exc).__name__}"
    elif HAS_AUTOGLUON:
        out["availability"]["autogluon"]["available"] = False
        out["availability"]["autogluon"]["reason"] = "row_limit_exceeded"
    if HAS_TABM and not CONFIG["enable_experimental_tabm_adapter"]:
        out["availability"]["tabm"]["reason"] = "installed_but_adapter_disabled"
    if HAS_TABR and not CONFIG["enable_experimental_tabr_adapter"]:
        out["availability"]["tabr"]["reason"] = "adapter_disabled"
    return out

def choose_winner(valid_metrics, metric, higher_is_better=False):
    usable = {k: v for k, v in valid_metrics.items() if isinstance(v, dict) and metric in v}
    if not usable:
        return None
    return max(usable, key=lambda k: usable[k][metric]) if higher_is_better else min(usable, key=lambda k: usable[k][metric])

def fit_tabular_regressor(train_x, y_train, valid_x, y_valid, objective="regression", alpha=None):
    print(f"  [Tabular] fitting regressors | objective={objective}" + (f" alpha={alpha}" if alpha else "") + f" | rows={len(train_x):,}")
    candidates = {}
    if HAS_LGB:
        params, hpo_info = tune_lgbm_regressor_params(train_x, y_train, valid_x, y_valid, objective, alpha)
        m = lgb.LGBMRegressor(**params)
        m.fit(train_x, y_train, eval_set=[(valid_x, y_valid)], callbacks=[lgb.early_stopping(80, verbose=False)])
        m.pios_hpo_info_ = hpo_info
        candidates["lightgbm"] = m
    if HAS_XGB and objective != "quantile":
        kwargs = dict(n_estimators=1500, learning_rate=.03, max_depth=8, subsample=.9, colsample_bytree=.9,
                      reg_lambda=1.2, random_state=SEED, n_jobs=-1, objective="reg:squarederror")
        if torch.cuda.is_available():
            kwargs.update(tree_method="hist", device="cuda")
        m = xgb.XGBRegressor(**kwargs)
        m.fit(train_x, y_train, eval_set=[(valid_x, y_valid)], verbose=False)
        candidates["xgboost"] = m
    if HAS_CAT and objective != "quantile":
        m = CatBoostRegressor(iterations=1500, depth=8, learning_rate=.03, loss_function="RMSE", verbose=False,
                              task_type="GPU" if torch.cuda.is_available() else "CPU", random_seed=SEED)
        try:
            m.fit(train_x, y_train, eval_set=(valid_x, y_valid), verbose=False)
            candidates["catboost"] = m
        except Exception as e:
            print("CatBoost skipped:", e)
    if not candidates:
        m = HistGradientBoostingRegressor(max_iter=800, learning_rate=.035, l2_regularization=.1, random_state=SEED)
        m.fit(train_x, y_train)
        candidates["hist_gbr"] = m
    best = min(candidates.items(), key=lambda kv: rmse(y_valid, kv[1].predict(valid_x)))
    print(f"  [Tabular] best={best[0]} | val_rmse={rmse(y_valid, best[1].predict(valid_x)):.4f}")
    return best[0], best[1], candidates

def fit_tabular_classifier(train_x, y_train, valid_x, y_valid):
    print(f"  [Tabular] fitting classifiers | rows={len(train_x):,} | pos_rate={np.mean(y_train):.3f}")
    candidates = {}
    pos = max(float(np.sum(np.asarray(y_train) == 1)), 1.0)
    neg = max(float(np.sum(np.asarray(y_train) == 0)), 1.0)
    spw = neg / pos
    if HAS_LGB:
        params, hpo_info = tune_lgbm_classifier_params(train_x, y_train, valid_x, y_valid, spw)
        m = lgb.LGBMClassifier(**params)
        m.fit(train_x, y_train, eval_set=[(valid_x, y_valid)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False)])
        m.pios_hpo_info_ = hpo_info
        candidates["lightgbm"] = m
    if HAS_XGB:
        kwargs = dict(n_estimators=1500, learning_rate=.03, max_depth=8, subsample=.9, colsample_bytree=.9,
                      reg_lambda=1.2, random_state=SEED, n_jobs=-1, objective="binary:logistic",
                      eval_metric="logloss", scale_pos_weight=spw)
        if torch.cuda.is_available():
            kwargs.update(tree_method="hist", device="cuda")
        m = xgb.XGBClassifier(**kwargs)
        m.fit(train_x, y_train, eval_set=[(valid_x, y_valid)], verbose=False)
        candidates["xgboost"] = m
    if HAS_CAT:
        m = CatBoostClassifier(iterations=1500, depth=8, learning_rate=.03, loss_function="Logloss", verbose=False,
                               task_type="GPU" if torch.cuda.is_available() else "CPU", random_seed=SEED,
                               scale_pos_weight=spw)
        try:
            m.fit(train_x, y_train, eval_set=(valid_x, y_valid), verbose=False)
            candidates["catboost"] = m
        except Exception as e:
            print("CatBoost skipped:", e)
    if not candidates:
        m = HistGradientBoostingClassifier(max_iter=800, learning_rate=.035, l2_regularization=.1, random_state=SEED)
        m.fit(train_x, y_train)
        candidates["hist_gbc"] = m

    def score(model):
        p = model.predict_proba(valid_x)[:, 1]
        return roc_auc_score(y_valid, p) if len(np.unique(y_valid)) > 1 else .5
    best = max(candidates.items(), key=lambda kv: score(kv[1]))
    print(f"  [Tabular] best={best[0]} | val_auc={score(best[1]):.4f}")
    return best[0], best[1], candidates


# Cell 5 — FT-Transformer (reduced for T4, no DataParallel, proper VRAM mgmt)

torch.backends.cudnn.benchmark = True


class NeuralPreprocessor:
    def __init__(self, feature_cols):
        self.feature_cols = list(dict.fromkeys(feature_cols))
        self.cat_cols, self.num_cols = [], []
        self.maps = {}
        self.scaler = StandardScaler()

    def fit(self, df):
        for c in self.feature_cols:
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
                self.cat_cols.append(c)
            else:
                self.num_cols.append(c)
        for c in self.cat_cols:
            vals = pd.Series(df[c].astype(str).fillna("__MISSING__").unique()).sort_values().tolist()
            self.maps[c] = {v: i+1 for i, v in enumerate(vals)}
        if self.num_cols:
            nums = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
            self.scaler.fit(nums)
        return self

    def transform(self, df):
        cats = []
        for c in self.cat_cols:
            cats.append(df[c].astype(str).fillna("__MISSING__").map(self.maps[c]).fillna(0).astype("int64").to_numpy())
        cat = np.vstack(cats).T.astype("int64") if cats else np.zeros((len(df), 0), dtype="int64")
        if self.num_cols:
            nums = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
            num = self.scaler.transform(nums).astype("float32")
        else:
            num = np.zeros((len(df), 0), dtype="float32")
        return cat, num

    @property
    def cardinalities(self):
        return [len(self.maps[c]) + 1 for c in self.cat_cols]


class NumericalPLR(nn.Module):
    def __init__(self, n_features, n_bins=CONFIG["n_plr_bins"], hidden=CONFIG["hidden_dim"]):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.edges = nn.Parameter(torch.linspace(0, 1, n_bins).unsqueeze(0).repeat(n_features, 1))
        self.proj = nn.Linear(n_bins, hidden)

    def forward(self, x):
        x_norm = torch.sigmoid(x)
        diff = x_norm.unsqueeze(-1) - self.edges.unsqueeze(0)
        plr = torch.clamp(diff, 0, 1/self.n_bins) * self.n_bins
        return self.proj(plr)


class ReGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out * 2)

    def forward(self, x):
        x, gate = self.linear(x).chunk(2, dim=-1)
        return x * F.relu(gate)


class FTTransformerBlock(nn.Module):
    def __init__(self, hidden, heads, dropout, ffn_mult):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        ffn_dim = int(hidden * ffn_mult)
        self.ffn = nn.Sequential(ReGLU(hidden, ffn_dim), nn.Dropout(dropout), nn.Linear(ffn_dim, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class FTTransformer(nn.Module):
    def __init__(self, cat_cards, n_num, out_dim, hidden=CONFIG["hidden_dim"],
                 heads=CONFIG["n_heads"], layers=CONFIG["n_transformer_layers"], dropout=CONFIG["dropout"]):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card, hidden) for card in cat_cards])
        self.use_plr = n_num > 0
        if self.use_plr:
            self.num_plr = NumericalPLR(n_num, CONFIG["n_plr_bins"], hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        ffn_mult = CONFIG["ffn_multiplier"]
        self.blocks = nn.ModuleList([FTTransformerBlock(hidden, heads, dropout, ffn_mult) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )
        nn.init.normal_(self.cls, std=.02)

    def forward(self, cat, num):
        toks = []
        for j, emb in enumerate(self.embs):
            toks.append(emb(cat[:, j]).unsqueeze(1))
        if self.use_plr:
            num_toks = self.num_plr(num)
            toks.append(num_toks)
        x = torch.cat([self.cls.expand(cat.shape[0], -1, -1)] + toks, dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])


def make_loader(cat, num, y, shuffle, batch_size=None):
    ds = TensorDataset(
        torch.tensor(cat, dtype=torch.long),
        torch.tensor(num, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    bs = int(batch_size or CONFIG["batch_size"])
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=CONFIG["neural_loader_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True, prefetch_factor=4
    )


def quantile_loss(pred, y, qs=torch.tensor([.1, .5, .9])):
    qs = qs.to(pred.device).view(1, -1)
    e = y.view(-1, 1) - pred
    return torch.mean(torch.maximum(qs*e, (qs-1)*e))


def _train_neural_single(train, valid, feature_cols, target, task, out_dim, prep, lr, dropout, hidden_dim, n_layers):
    trc, trn = prep.transform(train)
    vac, van = prep.transform(valid)
    ytr = train[target].astype(float).to_numpy()
    yva = valid[target].astype(float).to_numpy()
    model = FTTransformer(prep.cardinalities, trn.shape[1], out_dim=out_dim,
                          hidden=hidden_dim, layers=n_layers, dropout=dropout).to(DEVICE)
    if CONFIG.get("compile_model") and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    warmup_epochs = CONFIG["warmup_epochs"]
    total_epochs = CONFIG["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    train_loader = make_loader(trc, trn, ytr, True)
    valid_loader = make_loader(vac, van, yva, False, CONFIG["predict_batch_size"])
    accum_steps = CONFIG["gradient_accumulation_steps"]
    best_state, best_loss, bad = None, float("inf"), 0
    pos_weight = None
    if task == "classification":
        pos = max(np.sum(ytr == 1), 1)
        neg = max(np.sum(ytr == 0), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    epoch_bar = tqdm(range(CONFIG["epochs"]), desc="      Epochs", leave=False)
    for epoch in epoch_bar:
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (cb, nb_, yb) in enumerate(train_loader):
            cb = cb.to(DEVICE, non_blocking=True)
            nb_ = nb_.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(cb, nb_)
                if task == "quantile":
                    loss = quantile_loss(out, yb)
                elif task == "classification":
                    loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), yb, pos_weight=pos_weight)
                else:
                    loss = F.mse_loss(out.squeeze(-1), yb)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        scheduler.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for cb, nb_, yb in valid_loader:
                cb = cb.to(DEVICE)
                nb_ = nb_.to(DEVICE)
                yb = yb.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = model(cb, nb_)
                    if task == "quantile":
                        loss = quantile_loss(out, yb)
                    elif task == "classification":
                        loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), yb, pos_weight=pos_weight)
                    else:
                        loss = F.mse_loss(out.squeeze(-1), yb)
                losses.append(float(loss.cpu()))
        vl = float(np.mean(losses))
        if vl < best_loss:
            best_loss, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        epoch_bar.set_postfix(val_loss=f"{vl:.4f}", best=f"{best_loss:.4f}", patience=f"{bad}/{CONFIG['patience']}")
        if bad >= CONFIG["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {"best_valid_loss": best_loss, "epochs_ran": epoch + 1,
                   "lr": lr, "dropout": dropout, "hidden_dim": hidden_dim, "n_layers": n_layers}


def train_neural(train, valid, feature_cols, target, task="regression", out_dim=1):
    print(f"\n{'='*60}")
    print(f"Training FT-Transformer | target={target} | task={task} | out_dim={out_dim}")
    print(f"  train={len(train):,} | valid={len(valid):,} | features={len(feature_cols)}")
    print(f"{'='*60}")
    prep = NeuralPreprocessor(feature_cols).fit(train)

    if CONFIG.get("grid_search_enabled", False):
        grid = CONFIG["grid_search_neural"]
        combos = _grid_sample_combos(grid, CONFIG["grid_search_max_neural_combos"])
        print(f"  [Neural Grid Search] Testing {len(combos)} hyperparameter combos")
        best_model, best_info, best_loss = None, None, float("inf")
        for ci, combo in enumerate(combos):
            lr_c, drop_c = combo["lr"], combo["dropout"]
            hid_c, layers_c = combo["hidden_dim"], combo["n_transformer_layers"]
            print(f"    Combo {ci+1}/{len(combos)}: lr={lr_c} dropout={drop_c} hidden={hid_c} layers={layers_c}")
            model, info = _train_neural_single(train, valid, feature_cols, target, task, out_dim,
                                               prep, lr_c, drop_c, hid_c, layers_c)
            if info["best_valid_loss"] < best_loss:
                best_loss = info["best_valid_loss"]
                best_model, best_info = model, info
                print(f"      -> New best: val_loss={best_loss:.5f}")
            else:
                del model
            flush_gpu()
        print(f"  [Neural Grid Search] Best: val_loss={best_loss:.5f} | {best_info}")
        return best_model, prep, best_info

    print(f"  lr={CONFIG['lr']} | batch={CONFIG['batch_size']} | eff_batch={CONFIG['batch_size']*CONFIG['gradient_accumulation_steps']}")
    trc, trn = prep.transform(train)
    vac, van = prep.transform(valid)
    ytr = train[target].astype(float).to_numpy()
    yva = valid[target].astype(float).to_numpy()
    model = FTTransformer(prep.cardinalities, trn.shape[1], out_dim=out_dim).to(DEVICE)
    if CONFIG.get("compile_model") and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    warmup_epochs = CONFIG["warmup_epochs"]
    total_epochs = CONFIG["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    train_loader = make_loader(trc, trn, ytr, True)
    valid_loader = make_loader(vac, van, yva, False, CONFIG["predict_batch_size"])
    accum_steps = CONFIG["gradient_accumulation_steps"]
    best_state, best_loss, bad = None, float("inf"), 0
    pos_weight = None
    if task == "classification":
        pos = max(np.sum(ytr == 1), 1)
        neg = max(np.sum(ytr == 0), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    epoch_bar = tqdm(range(CONFIG["epochs"]), desc="  Epochs", leave=False)
    for epoch in epoch_bar:
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (cb, nb, yb) in enumerate(train_loader):
            cb, nb, yb = cb.to(DEVICE, non_blocking=True), nb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(cb, nb)
                if task == "quantile":
                    loss = quantile_loss(out, yb)
                elif task == "classification":
                    loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), yb, pos_weight=pos_weight)
                else:
                    loss = F.mse_loss(out.squeeze(-1), yb)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        scheduler.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for cb, nb, yb in valid_loader:
                cb, nb, yb = cb.to(DEVICE), nb.to(DEVICE), yb.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = model(cb, nb)
                    if task == "quantile":
                        loss = quantile_loss(out, yb)
                    elif task == "classification":
                        loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), yb, pos_weight=pos_weight)
                    else:
                        loss = F.mse_loss(out.squeeze(-1), yb)
                losses.append(float(loss.cpu()))
        vl = float(np.mean(losses))
        if vl < best_loss:
            best_loss, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        epoch_bar.set_postfix(val_loss=f"{vl:.5f}", best=f"{best_loss:.5f}", patience=f"{bad}/{CONFIG['patience']}")
        if bad >= CONFIG["patience"]:
            break
    model.load_state_dict(best_state)
    return model, prep, {"best_valid_loss": best_loss, "epochs_ran": epoch + 1}


def neural_predict(model, prep, df, task="regression"):
    cat, num = prep.transform(df)
    loader = make_loader(cat, num, np.zeros(len(df), dtype="float32"), False, CONFIG["predict_batch_size"])
    preds = []
    model.eval()
    with torch.no_grad():
        for cb, nb, _ in loader:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(cb.to(DEVICE), nb.to(DEVICE))
            if task == "classification":
                out = torch.sigmoid(out.squeeze(-1))
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

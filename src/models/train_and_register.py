"""
train_and_register.py
=====================
Phase 2 – Full model training, evaluation, quality gating, and W&B Model Registry push.

Models trained:
  Regression  (target: preparedness_score 0-100):
    - Ridge Regression (L2)
    - Random Forest Regressor
    - HistGradientBoosting Regressor
    - AdaBoost Regressor

  Classification (target: preparedness_tier Low/Medium/High):
    - Logistic Regression (Multinomial)
    - Random Forest Classifier
    - HistGradientBoosting Classifier
    - AdaBoost Classifier

  Unsupervised (exploratory tier structure validation):
    - DBSCAN (spatial cluster analysis on feature space)

Quality Gates (models MUST pass to be promoted to registry):
  - Regression champion:     R² >= 0.90, RMSE <= 10.0
  - Classification champion: Macro F1 >= 0.80, Accuracy >= 0.85

Outputs:
  - models/registry/<model_name>_<split>.joblib
  - models/registry/registry_manifest.json
  - W&B Model Registry artifacts (staged as "champion")
"""

import os
import sys
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, classification_report
)

# Regressors
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, RandomForestClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

# DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import wandb

# W&B is skipped when WANDB_DISABLED=true (sandbox / no-API-key environments).
# Models are still saved to models/registry/ locally.
# To push to W&B: set WANDB_API_KEY and re-run without WANDB_DISABLED.
USE_WANDB = os.environ.get("WANDB_DISABLED", "false").lower() != "true"
if not USE_WANDB:
    print("[INFO] WANDB_DISABLED=true — W&B logging skipped, models saved locally only.")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "modelling_dataset.parquet"
REGISTRY_DIR = ROOT / "models" / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = REGISTRY_DIR / "registry_manifest.json"

# ── Quality Gate Thresholds ───────────────────────────────────────────────────
QG_REG_R2_MIN    = 0.90
QG_REG_RMSE_MAX  = 10.0
QG_CLS_F1_MIN    = 0.80
QG_CLS_ACC_MIN   = 0.85

# ── Feature & Target Columns ─────────────────────────────────────────────────
MODEL_FEATURES = [
    "dist_coast_km", "elevation_mean", "avg_household_size",
    "sc_share", "st_share", "vulnerable_pop_share",
    "road_connectivity_score", "drinking_water_score",
    "n_hc_10km", "n_phc_chc_10km", "n_gauges_20km", "n_shelters_10km",
    "cvi_physical", "cvi_social",
    "has_pucca_road", "has_major_district_road", "bus_connectivity",
    "log1p_population", "log1p_pop_density_sqkm",
    "log1p_dist_nearest_shelter_km", "log1p_dist_nearest_hospital_km",
    "coastal_risk_score", "evacuation_sufficiency",
    "vulnerability_exposure", "infra_composite", "monitoring_coverage",
]

TARGET_REG = "preparedness_score"
TARGET_CLS = "preparedness_tier"

SPLIT_CONFIGS = [
    {"train": 0.60, "val": 0.20, "test": 0.20, "label": "60_20_20"},
    {"train": 0.70, "val": 0.15, "test": 0.15, "label": "70_15_15"},
    {"train": 0.80, "val": 0.10, "test": 0.10, "label": "80_10_10"},
]

WANDB_PROJECT = "cyclone-preparedness-phase2"
WANDB_ENTITY  = None   # set via WANDB_ENTITY env var or leave None for personal

# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print(f"\nLoading modelling dataset from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    # Build engineered features if missing (idempotent)
    if "log1p_population" not in df.columns:
        for feat in ["population", "pop_density_sqkm", "dist_nearest_hospital_km",
                     "dist_nearest_shelter_km", "shelter_capacity_nearby",
                     "shelter_capacity_per_capita"]:
            if feat in df.columns:
                df[f"log1p_{feat}"] = np.log1p(df[feat].clip(0))

    if "coastal_risk_score" not in df.columns:
        df["coastal_risk_score"] = (
            np.exp(-df["dist_coast_km"] / 10.0) *
            np.exp(-df["elevation_mean"] / 30.0)
        ).round(4)

    if "evacuation_sufficiency" not in df.columns:
        pop_safe = df["population"].replace(0, np.nan)
        df["evacuation_sufficiency"] = (
            df["shelter_capacity_nearby"] / pop_safe.fillna(1)
        ).clip(0, 10).round(4)

    if "vulnerability_exposure" not in df.columns:
        df["vulnerability_exposure"] = (
            0.5 * (df["cvi_integrated"] / df["cvi_integrated"].max()) +
            0.3 * (1 - df["dist_coast_km"] / df["dist_coast_km"].max()) +
            0.2 * df["vulnerable_pop_share"]
        ).round(4)

    if "infra_composite" not in df.columns:
        df["infra_composite"] = (
            df["road_connectivity_score"] / 3.0 +
            df["drinking_water_score"] / 2.0 +
            df["bus_connectivity"].astype(float)
        ).round(4) / 3.0

    if "monitoring_coverage" not in df.columns:
        mx = df["dist_nearest_gauge_km"].max()
        mn = df["n_gauges_20km"].max()
        df["monitoring_coverage"] = (
            (1 - df["dist_nearest_gauge_km"] / mx) * 0.6 +
            (df["n_gauges_20km"] / mn) * 0.4
        ).round(4)

    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing   = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (will be skipped): {missing}")

    X = df[available].fillna(0)
    y_reg = df[TARGET_REG]
    y_cls_raw = df[TARGET_CLS].astype(str)

    le = LabelEncoder()
    y_cls = le.fit_transform(y_cls_raw)

    print(f"  Dataset shape : {X.shape}")
    print(f"  Target range  : {y_reg.min():.1f} – {y_reg.max():.1f}")
    print(f"  Class dist    : {pd.Series(y_cls_raw).value_counts().to_dict()}")
    return X, y_reg, y_cls, le, available


def make_regression_pipelines():
    return {
        "ridge": Pipeline([
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=10.0))
        ]),
        "random_forest_reg": Pipeline([
            ("scaler", RobustScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=12,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ))
        ]),
        "histgbt_reg": Pipeline([
            ("scaler", RobustScaler()),
            ("model", HistGradientBoostingRegressor(
                max_iter=200, max_depth=5,
                learning_rate=0.05, min_samples_leaf=20, random_state=42
            ))
        ]),
        "adaboost_reg": Pipeline([
            ("scaler", RobustScaler()),
            ("model", AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4),
                n_estimators=150, learning_rate=0.1, random_state=42
            ))
        ]),
    }


def make_classification_pipelines():
    return {
        "logistic_reg": Pipeline([
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(
                max_iter=1000, C=1.0, random_state=42,
                solver="lbfgs"
            ))
        ]),
        "random_forest_cls": Pipeline([
            ("scaler", RobustScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_leaf=5, class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ]),
        "histgbt_cls": Pipeline([
            ("scaler", RobustScaler()),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, max_depth=5,
                learning_rate=0.05, min_samples_leaf=20, random_state=42
            ))
        ]),
        "adaboost_cls": Pipeline([
            ("scaler", RobustScaler()),
            ("model", AdaBoostClassifier(
                n_estimators=150, learning_rate=0.1,
                random_state=42
            ))
        ]),
    }


def run_quality_gate(task, metrics):
    """Returns (passed: bool, reason: str)"""
    if task == "regression":
        r2   = metrics.get("test_r2", 0)
        rmse = metrics.get("test_rmse", 999)
        if r2 < QG_REG_R2_MIN:
            return False, f"R2 {r2:.3f} < threshold {QG_REG_R2_MIN}"
        if rmse > QG_REG_RMSE_MAX:
            return False, f"RMSE {rmse:.3f} > threshold {QG_REG_RMSE_MAX}"
        return True, f"R2={r2:.3f}, RMSE={rmse:.3f}"
    else:
        f1  = metrics.get("test_f1_macro", 0)
        acc = metrics.get("test_accuracy", 0)
        if f1 < QG_CLS_F1_MIN:
            return False, f"Macro F1 {f1:.3f} < threshold {QG_CLS_F1_MIN}"
        if acc < QG_CLS_ACC_MIN:
            return False, f"Accuracy {acc:.3f} < threshold {QG_CLS_ACC_MIN}"
        return True, f"F1={f1:.3f}, Acc={acc:.3f}"


def save_to_registry(pipe, model_name, split_label, task, metrics, passed_qg, le=None):
    fname = REGISTRY_DIR / f"{model_name}_{split_label}.joblib"
    artifact_dict = {"pipeline": pipe}
    if le is not None:
        artifact_dict["label_encoder"] = le
    joblib.dump(artifact_dict, fname)

    entry = {
        "model_name"  : model_name,
        "split_label" : split_label,
        "task"        : task,
        "metrics"     : metrics,
        "quality_gate": "PASS" if passed_qg else "FAIL",
        "file"        : str(fname.relative_to(ROOT)),
        "saved_at"    : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return entry, str(fname)


def push_to_wandb_registry(run, model_name, split_label, task, metrics,
                           passed_qg, filepath, le=None):
    """Log model artifact to W&B and add to W&B Model Registry."""
    artifact_name = f"{model_name}_{split_label}"
    art = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"{task} model — {model_name} trained on {split_label} split",
        metadata={
            **metrics,
            "quality_gate": "PASS" if passed_qg else "FAIL",
            "task": task,
            "split": split_label,
        }
    )
    art.add_file(filepath)
    if le is not None:
        # Save label encoder separately
        le_path = str(filepath).replace(".joblib", "_le.joblib")
        joblib.dump(le, le_path)
        art.add_file(le_path)

    aliases = ["latest"]
    if passed_qg:
        aliases.append("champion")

    run.log_artifact(art, aliases=aliases)
    print(f"    W&B artifact pushed: {artifact_name} | aliases={aliases}")


def run_dbscan_analysis(X, y_reg, features_used):
    """DBSCAN exploratory cluster analysis to validate tier structure."""
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTER ANALYSIS (Exploratory)")
    print("=" * 60)

    # Reduce to 2D via PCA for clustering stability
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    db = DBSCAN(eps=1.5, min_samples=15, metric="euclidean")
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({100*n_noise/len(labels):.1f}%)")

    for c in sorted(set(labels)):
        mask = labels == c
        label = "Noise" if c == -1 else f"Cluster {c}"
        mean_score = y_reg[mask].mean()
        count = mask.sum()
        print(f"  {label:12s}: n={count:4d}, mean preparedness_score={mean_score:.1f}")

    result = {
        "n_clusters"       : n_clusters,
        "n_noise"          : n_noise,
        "noise_pct"        : round(100 * n_noise / len(labels), 2),
        "pca_variance_expl": round(float(pca.explained_variance_ratio_.sum()), 4),
    }

    # Log cluster analysis to W&B
    if USE_WANDB:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name="dbscan_cluster_analysis",
                config={"eps": 1.5, "min_samples": 15, "pca_components": 10},
                reinit=True,
            )
            wandb.log(result)
            wandb.finish()
            print("  W&B run logged: dbscan_cluster_analysis")
        except Exception as e:
            print(f"  [WARN] W&B DBSCAN log failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 2 — TRAIN, EVALUATE, QUALITY GATE & REGISTER MODELS")
    print("=" * 70)

    X, y_reg, y_cls, le, features_used = load_data()

    all_registry_entries = []
    best_reg = {"r2": -999, "entry": None}
    best_cls = {"f1": -999, "entry": None}

    # ── REGRESSION ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REGRESSION MODELS")
    print("=" * 60)

    for split_cfg in SPLIT_CONFIGS:
        label = split_cfg["label"]
        test_sz = split_cfg["test"]
        val_sz  = split_cfg["val"] / (1 - test_sz)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_reg, test_size=test_sz, random_state=42
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=val_sz, random_state=42
        )

        for model_name, pipe in make_regression_pipelines().items():
            print(f"\n  [{label}] {model_name}")
            pipe.fit(X_tr, y_tr)

            metrics = {
                "train_r2"  : round(r2_score(y_tr,  pipe.predict(X_tr)),  4),
                "val_r2"    : round(r2_score(y_val, pipe.predict(X_val)), 4),
                "test_r2"   : round(r2_score(y_te,  pipe.predict(X_te)),  4),
                "train_rmse": round(np.sqrt(mean_squared_error(y_tr,  pipe.predict(X_tr))),  4),
                "val_rmse"  : round(np.sqrt(mean_squared_error(y_val, pipe.predict(X_val))), 4),
                "test_rmse" : round(np.sqrt(mean_squared_error(y_te,  pipe.predict(X_te))),  4),
                "test_mae"  : round(mean_absolute_error(y_te, pipe.predict(X_te)), 4),
            }

            passed, reason = run_quality_gate("regression", metrics)
            status = "PASS" if passed else "FAIL"
            print(f"    QG {status}: R2={metrics['test_r2']:.3f}, RMSE={metrics['test_rmse']:.3f} | {reason}")

            # Log to W&B
            if USE_WANDB:
                try:
                    run = wandb.init(
                        project=WANDB_PROJECT,
                        entity=WANDB_ENTITY,
                        name=f"reg_{model_name}_{label}",
                        config={
                            "task": "regression", "model": model_name,
                            "split": label, "features": len(features_used),
                            "quality_gate_threshold_r2": QG_REG_R2_MIN,
                            "quality_gate_threshold_rmse": QG_REG_RMSE_MAX,
                        },
                        reinit=True,
                    )
                    wandb.log({**metrics, "quality_gate_passed": int(passed)})
                    entry, fpath = save_to_registry(
                        pipe, model_name, label, "regression", metrics, passed
                    )
                    push_to_wandb_registry(run, model_name, label, "regression",
                                           metrics, passed, fpath)
                    wandb.finish()
                except Exception as e:
                    print(f"    [WARN] W&B log failed: {e}")
                    entry, fpath = save_to_registry(
                        pipe, model_name, label, "regression", metrics, passed
                    )
            else:
                entry, fpath = save_to_registry(
                    pipe, model_name, label, "regression", metrics, passed
                )

            all_registry_entries.append(entry)

            if passed and metrics["test_r2"] > best_reg["r2"]:
                best_reg = {"r2": metrics["test_r2"], "entry": entry}

    # ── CLASSIFICATION ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION MODELS")
    print("=" * 60)

    for split_cfg in SPLIT_CONFIGS:
        label = split_cfg["label"]
        test_sz = split_cfg["test"]
        val_sz  = split_cfg["val"] / (1 - test_sz)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_cls, test_size=test_sz, random_state=42, stratify=y_cls
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=val_sz, random_state=42, stratify=y_tr
        )

        for model_name, pipe in make_classification_pipelines().items():
            print(f"\n  [{label}] {model_name}")
            pipe.fit(X_tr, y_tr)

            metrics = {
                "train_accuracy": round(accuracy_score(y_tr,  pipe.predict(X_tr)),  4),
                "val_accuracy"  : round(accuracy_score(y_val, pipe.predict(X_val)), 4),
                "test_accuracy" : round(accuracy_score(y_te,  pipe.predict(X_te)),  4),
                "train_f1_macro": round(f1_score(y_tr,  pipe.predict(X_tr),  average="macro"), 4),
                "val_f1_macro"  : round(f1_score(y_val, pipe.predict(X_val), average="macro"), 4),
                "test_f1_macro" : round(f1_score(y_te,  pipe.predict(X_te),  average="macro"), 4),
                "test_f1_weighted": round(f1_score(y_te, pipe.predict(X_te), average="weighted"), 4),
            }

            passed, reason = run_quality_gate("classification", metrics)
            status = "PASS" if passed else "FAIL"
            print(f"    QG {status}: Acc={metrics['test_accuracy']:.3f}, F1={metrics['test_f1_macro']:.3f} | {reason}")

            if USE_WANDB:
                try:
                    run = wandb.init(
                        project=WANDB_PROJECT,
                        entity=WANDB_ENTITY,
                        name=f"cls_{model_name}_{label}",
                        config={
                            "task": "classification", "model": model_name,
                            "split": label, "features": len(features_used),
                            "quality_gate_threshold_f1": QG_CLS_F1_MIN,
                            "quality_gate_threshold_acc": QG_CLS_ACC_MIN,
                        },
                        reinit=True,
                    )
                    wandb.log({**metrics, "quality_gate_passed": int(passed)})
                    entry, fpath = save_to_registry(
                        pipe, model_name, label, "classification", metrics, passed, le
                    )
                    push_to_wandb_registry(run, model_name, label, "classification",
                                           metrics, passed, fpath, le)
                    wandb.finish()
                except Exception as e:
                    print(f"    [WARN] W&B log failed: {e}")
                    entry, fpath = save_to_registry(
                        pipe, model_name, label, "classification", metrics, passed, le
                    )
            else:
                entry, fpath = save_to_registry(
                    pipe, model_name, label, "classification", metrics, passed, le
                )

            all_registry_entries.append(entry)

            if passed and metrics["test_f1_macro"] > best_cls["f1"]:
                best_cls = {"f1": metrics["test_f1_macro"], "entry": entry}

    # ── DBSCAN ──────────────────────────────────────────────────────────────
    dbscan_result = run_dbscan_analysis(X, y_reg, features_used)

    # ── MANIFEST ────────────────────────────────────────────────────────────
    manifest = {
        "generated_at"       : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_runs"         : len(all_registry_entries),
        "quality_gate_thresholds": {
            "regression_r2_min"   : QG_REG_R2_MIN,
            "regression_rmse_max" : QG_REG_RMSE_MAX,
            "classification_f1_min"  : QG_CLS_F1_MIN,
            "classification_acc_min" : QG_CLS_ACC_MIN,
        },
        "champion_regression"    : best_reg["entry"],
        "champion_classification": best_cls["entry"],
        "dbscan_analysis"        : dbscan_result,
        "all_runs"               : all_registry_entries,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("REGISTRY MANIFEST WRITTEN:")
    print(f"  {MANIFEST_PATH}")

    if best_reg["entry"]:
        e = best_reg["entry"]
        print(f"\nCHAMPION REGRESSOR:     {e['model_name']} [{e['split_label']}]")
        print(f"  Test R2  = {e['metrics']['test_r2']:.4f}")
        print(f"  Test RMSE= {e['metrics']['test_rmse']:.4f}")

    if best_cls["entry"]:
        e = best_cls["entry"]
        print(f"\nCHAMPION CLASSIFIER:    {e['model_name']} [{e['split_label']}]")
        print(f"  Test F1  = {e['metrics']['test_f1_macro']:.4f}")
        print(f"  Test Acc = {e['metrics']['test_accuracy']:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()

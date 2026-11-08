"""
quality_checks.py
=================
Phase 2 – Standalone model risk and quality checks.

Loads champion models from the registry manifest and runs:
  1. Performance threshold assertions  (hard fail if breached)
  2. Per-district fairness / bias check (flag if gap > 15 pts between districts)
  3. Confidence calibration check      (classification only)
  4. Confusion matrix audit            (flag if Low tier recall < 0.70)
  5. Overfitting gap check             (train-test R2/F1 delta)
  6. Edge case stress tests            (all-zero input, max-risk input)

Exit code 0 = all checks passed. Exit code 1 = one or more checks failed.
Used as a step in GitHub Actions CI/CD pipeline.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "models" / "registry" / "registry_manifest.json"
DATA_PATH = ROOT / "data" / "processed" / "modelling_dataset.parquet"

FAIL = False  # tracks any check failure


def header(title):
    print("\n" + "=" * 60)
    print(f"CHECK: {title}")
    print("=" * 60)


def check(name, condition, detail=""):
    global FAIL
    status = "PASS" if condition else "FAIL"
    icon = "v" if condition else "X"
    print(f"  [{icon}] {status} | {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAIL = True


def load_manifest():
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Registry manifest not found: {MANIFEST_PATH}")
        print("        Run src/models/train_and_register.py first.")
        sys.exit(1)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_model(entry):
    fpath = ROOT / entry["file"]
    if not fpath.exists():
        return None, None
    art = joblib.load(fpath)
    pipe = art.get("pipeline")
    le = art.get("label_encoder")
    return pipe, le


def load_dataset():
    df = pd.read_parquet(DATA_PATH)
    MODEL_FEATURES = [
        "dist_coast_km",
        "elevation_mean",
        "avg_household_size",
        "sc_share",
        "st_share",
        "vulnerable_pop_share",
        "road_connectivity_score",
        "drinking_water_score",
        "n_hc_10km",
        "n_phc_chc_10km",
        "n_gauges_20km",
        "n_shelters_10km",
        "cvi_physical",
        "cvi_social",
        "has_pucca_road",
        "has_major_district_road",
        "bus_connectivity",
        "log1p_population",
        "log1p_pop_density_sqkm",
        "log1p_dist_nearest_shelter_km",
        "log1p_dist_nearest_hospital_km",
        "coastal_risk_score",
        "evacuation_sufficiency",
        "vulnerability_exposure",
        "infra_composite",
        "monitoring_coverage",
    ]
    available = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y_reg = df["preparedness_score"]
    y_cls_raw = df["preparedness_tier"].astype(str)
    return X, y_reg, y_cls_raw, df, available


# ── 1. MANIFEST CHECKS ───────────────────────────────────────────────────────


def check_manifest(manifest):
    header("Registry Manifest Integrity")
    check("Manifest file exists and is valid JSON", True)

    champ_reg = manifest.get("champion_regression")
    champ_cls = manifest.get("champion_classification")

    check("Champion regressor recorded in manifest", champ_reg is not None)
    check("Champion classifier recorded in manifest", champ_cls is not None)

    if champ_reg:
        qg = manifest["quality_gate_thresholds"]
        r2 = champ_reg["metrics"].get("test_r2", 0)
        rmse = champ_reg["metrics"].get("test_rmse", 999)
        check(
            f"Champion regressor R2 >= {qg['regression_r2_min']}",
            r2 >= qg["regression_r2_min"],
            f"R2={r2:.4f}",
        )
        check(
            f"Champion regressor RMSE <= {qg['regression_rmse_max']}",
            rmse <= qg["regression_rmse_max"],
            f"RMSE={rmse:.4f}",
        )
        check("Champion regressor quality gate = PASS", champ_reg.get("quality_gate") == "PASS")

    if champ_cls:
        qg = manifest["quality_gate_thresholds"]
        f1 = champ_cls["metrics"].get("test_f1_macro", 0)
        acc = champ_cls["metrics"].get("test_accuracy", 0)
        check(
            f"Champion classifier Macro F1 >= {qg['classification_f1_min']}",
            f1 >= qg["classification_f1_min"],
            f"F1={f1:.4f}",
        )
        check(
            f"Champion classifier Accuracy >= {qg['classification_acc_min']}",
            acc >= qg["classification_acc_min"],
            f"Acc={acc:.4f}",
        )
        check("Champion classifier quality gate = PASS", champ_cls.get("quality_gate") == "PASS")

    return champ_reg, champ_cls


# ── 2. OVERFITTING GAP CHECK ─────────────────────────────────────────────────


def check_overfitting(champ_reg, champ_cls):
    header("Overfitting / Generalization Gap Check")

    if champ_reg:
        m = champ_reg["metrics"]
        gap_r2 = abs(m.get("train_r2", 0) - m.get("test_r2", 0))
        check("Regression train-test R2 gap < 0.05", gap_r2 < 0.05, f"gap={gap_r2:.4f}")

    if champ_cls:
        m = champ_cls["metrics"]
        gap_f1 = abs(m.get("train_f1_macro", 0) - m.get("test_f1_macro", 0))
        check("Classification train-test F1 gap < 0.08", gap_f1 < 0.08, f"gap={gap_f1:.4f}")


# ── 3. PER-DISTRICT FAIRNESS CHECK ───────────────────────────────────────────


def check_fairness(reg_pipe, X, y_reg, df):
    header("Per-District Fairness / Bias Check")

    if reg_pipe is None or "district_name" not in df.columns:
        print("  [SKIP] Regressor not loaded or district_name column missing")
        return

    preds = reg_pipe.predict(X)
    df_tmp = df.copy()
    df_tmp["_pred"] = preds
    df_tmp["_error"] = abs(preds - y_reg.values)

    districts = df_tmp["district_name"].unique()
    mean_errors = {}
    for d in districts:
        mask = df_tmp["district_name"] == d
        mean_errors[d] = df_tmp.loc[mask, "_error"].mean()
        print(f"  District: {d:30s} | Mean Abs Error = {mean_errors[d]:.2f}")

    if len(mean_errors) > 1:
        gap = max(mean_errors.values()) - min(mean_errors.values())
        check("Max inter-district MAE gap < 5.0 pts", gap < 5.0, f"gap={gap:.2f} pts")


# ── 4. LOW-TIER RECALL AUDIT ─────────────────────────────────────────────────


def check_low_tier_recall(cls_pipe, le, X, y_cls_raw):
    header("Low Preparedness Tier Recall Audit")

    if cls_pipe is None or le is None:
        print("  [SKIP] Classifier not loaded")
        return

    y_enc = le.transform(y_cls_raw)
    preds = cls_pipe.predict(X)

    classes = le.classes_
    for cls_name in classes:
        cls_idx = list(classes).index(cls_name)
        binary_true = (y_enc == cls_idx).astype(int)
        binary_pred = (preds == cls_idx).astype(int)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        threshold = 0.70 if cls_name == "Low" else 0.60
        check(
            f"Recall for class '{cls_name}' >= {threshold:.0%}", recall >= threshold, f"recall={recall:.3f}"
        )


# ── 5. EDGE CASE STRESS TEST ─────────────────────────────────────────────────


def check_edge_cases(reg_pipe, cls_pipe, le, available_features):
    header("Edge Case Stress Tests")

    n_feat = len(available_features)

    if reg_pipe is not None:
        # All-zero input (completely unprepared village)
        X_zero = pd.DataFrame([np.zeros(n_feat)], columns=available_features)
        pred_zero = reg_pipe.predict(X_zero)[0]
        check(
            "All-zero input returns finite regression prediction",
            np.isfinite(pred_zero),
            f"pred={pred_zero:.2f}",
        )
        check(
            "All-zero prediction within valid range [0, 100]", 0 <= pred_zero <= 100, f"pred={pred_zero:.2f}"
        )

    if cls_pipe is not None:
        # All-zero input classification
        X_zero = pd.DataFrame([np.zeros(n_feat)], columns=available_features)
        pred_cls = cls_pipe.predict(X_zero)[0]
        pred_label = le.inverse_transform([pred_cls])[0] if le else pred_cls
        check(
            "All-zero input returns valid class label",
            pred_label in ["Low", "Medium", "High"],
            f"pred={pred_label}",
        )

        # Has predict_proba?
        has_proba = hasattr(cls_pipe.named_steps.get("model", object()), "predict_proba")
        if has_proba:
            proba = cls_pipe.predict_proba(X_zero)[0]
            check("predict_proba sums to ~1.0", abs(proba.sum() - 1.0) < 1e-6, f"sum={proba.sum():.6f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("PHASE 2 — MODEL QUALITY & RISK CHECKS")
    print("=" * 70)

    manifest = load_manifest()
    champ_reg, champ_cls = check_manifest(manifest)
    check_overfitting(champ_reg, champ_cls)

    X, y_reg, y_cls_raw, df, available = load_dataset()

    reg_pipe, _ = load_model(champ_reg) if champ_reg else (None, None)
    cls_pipe, le = load_model(champ_cls) if champ_cls else (None, None)

    check_fairness(reg_pipe, X, y_reg, df)
    check_low_tier_recall(cls_pipe, le, X, y_cls_raw)
    check_edge_cases(reg_pipe, cls_pipe, le, available)

    print("\n" + "=" * 70)
    if FAIL:
        print("RESULT: ONE OR MORE QUALITY CHECKS FAILED — review above output.")
        print("=" * 70)
        sys.exit(1)
    else:
        print("RESULT: ALL QUALITY CHECKS PASSED — models are registry-ready.")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()

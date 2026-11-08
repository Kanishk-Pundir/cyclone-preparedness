"""
app/main.py
===========
Phase 2 – FastAPI Inference Service

Endpoints:
  GET  /health              — liveness + model info
  GET  /model-info          — registry champion metadata
  POST /predict             — single village regression + classification
  POST /predict/batch       — batch inference (list of villages)
  GET  /predict/sample      — sample prediction using test village data

The service loads champion models from models/registry/registry_manifest.json
at startup. No scikit-learn retraining at runtime.
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "registry" / "registry_manifest.json"

# ── Feature list (must match training) ───────────────────────────────────────
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


# ── App State ─────────────────────────────────────────────────────────────────
class AppState:
    reg_pipe = None
    cls_pipe = None
    le = None
    manifest = None
    startup_time = None
    available_features: List[str] = []


state = AppState()

# ── Lifespan Context Manager: Load Models on Startup ────────────────────────


def _load_champion_models():
    state.startup_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    if not MANIFEST_PATH.exists():
        print(
            f"[WARN] Registry manifest not found at {MANIFEST_PATH}. "
            f"Endpoints will return 503 until models are trained."
        )
        return

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        state.manifest = json.load(f)

    def _load(entry):
        if entry is None:
            return None, None
        fpath = ROOT / entry["file"]
        if not fpath.exists():
            return None, None
        art = joblib.load(fpath)
        return art.get("pipeline"), art.get("label_encoder")

    state.reg_pipe, _ = _load(state.manifest.get("champion_regression"))
    state.cls_pipe, state.le = _load(state.manifest.get("champion_classification"))
    state.available_features = MODEL_FEATURES

    reg_name = state.manifest.get("champion_regression", {}).get("model_name", "none")
    cls_name = state.manifest.get("champion_classification", {}).get("model_name", "none")
    print(f"[OK] Models loaded. Regressor: {reg_name} | Classifier: {cls_name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_champion_models()
    yield


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cyclone Preparedness Inference API",
    description=(
        "AI-Based Disaster Preparedness Assistant — Phase 2 Inference Service.\n\n"
        "Predicts village-level cyclone preparedness score (0-100) and "
        "preparedness tier (Low / Medium / High) for coastal Karnataka villages."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class VillageFeatures(BaseModel):
    # Hazard Exposure
    dist_coast_km: float = Field(..., ge=0, le=200, description="Distance to coastline (km)")
    elevation_mean: float = Field(..., ge=0, le=1000, description="Mean elevation (m)")

    # Demographics
    avg_household_size: float = Field(4.5, ge=1, le=15)
    sc_share: float = Field(0.0, ge=0, le=1, description="Scheduled Caste population fraction")
    st_share: float = Field(0.0, ge=0, le=1, description="Scheduled Tribe population fraction")
    vulnerable_pop_share: float = Field(0.0, ge=0, le=1)

    # Infrastructure
    road_connectivity_score: float = Field(0.0, ge=0, le=3)
    drinking_water_score: float = Field(0.0, ge=0, le=2)
    has_pucca_road: int = Field(0, ge=0, le=1)
    has_major_district_road: int = Field(0, ge=0, le=1)
    bus_connectivity: int = Field(0, ge=0, le=1)

    # Health facilities
    n_hc_10km: int = Field(0, ge=0)
    n_phc_chc_10km: int = Field(0, ge=0)

    # Weather monitoring
    n_gauges_20km: int = Field(0, ge=0)

    # Shelters
    n_shelters_10km: int = Field(0, ge=0)

    # Coastal Vulnerability Index
    cvi_physical: float = Field(3.0, ge=1, le=5)
    cvi_social: float = Field(3.0, ge=1, le=5)

    # Engineered features (optional — computed server-side if not provided)
    log1p_population: Optional[float] = None
    log1p_pop_density_sqkm: Optional[float] = None
    log1p_dist_nearest_shelter_km: Optional[float] = None
    log1p_dist_nearest_hospital_km: Optional[float] = None
    coastal_risk_score: Optional[float] = None
    evacuation_sufficiency: Optional[float] = None
    vulnerability_exposure: Optional[float] = None
    infra_composite: Optional[float] = None
    monitoring_coverage: Optional[float] = None

    # Optional raw inputs to compute engineered features
    population: Optional[float] = Field(None, ge=0)
    pop_density_sqkm: Optional[float] = Field(None, ge=0)
    dist_nearest_shelter_km: Optional[float] = Field(None, ge=0)
    dist_nearest_hospital_km: Optional[float] = Field(None, ge=0)
    shelter_capacity_nearby: Optional[float] = Field(None, ge=0)
    cvi_integrated: Optional[float] = Field(None, ge=1, le=5)

    class Config:
        json_schema_extra = {
            "example": {
                "dist_coast_km": 2.5,
                "elevation_mean": 8.0,
                "avg_household_size": 4.2,
                "sc_share": 0.18,
                "st_share": 0.05,
                "vulnerable_pop_share": 0.23,
                "road_connectivity_score": 1.0,
                "drinking_water_score": 1.0,
                "has_pucca_road": 1,
                "has_major_district_road": 0,
                "bus_connectivity": 1,
                "n_hc_10km": 8,
                "n_phc_chc_10km": 2,
                "n_gauges_20km": 15,
                "n_shelters_10km": 1,
                "cvi_physical": 4.0,
                "cvi_social": 3.5,
                "population": 1200,
                "dist_nearest_shelter_km": 3.2,
                "dist_nearest_hospital_km": 42.0,
                "shelter_capacity_nearby": 800,
                "cvi_integrated": 3.8,
            }
        }


class PredictionResponse(BaseModel):
    preparedness_score: float
    preparedness_tier: str
    tier_probabilities: Optional[Dict[str, float]]
    risk_level: str
    interpretation: str
    model_info: Dict[str, str]


class BatchRequest(BaseModel):
    villages: List[VillageFeatures]


class BatchResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


# ── Helper: Feature Engineering ───────────────────────────────────────────────
def engineer_features(v: VillageFeatures) -> pd.DataFrame:
    """Compute engineered features server-side if not provided."""
    d = v.dict()

    if d.get("log1p_population") is None:
        pop = d.get("population") or 0
        d["log1p_population"] = float(np.log1p(max(0, pop)))

    if d.get("log1p_pop_density_sqkm") is None:
        dens = d.get("pop_density_sqkm") or 0
        d["log1p_pop_density_sqkm"] = float(np.log1p(max(0, dens)))

    if d.get("log1p_dist_nearest_shelter_km") is None:
        ds = d.get("dist_nearest_shelter_km") or 0
        d["log1p_dist_nearest_shelter_km"] = float(np.log1p(max(0, ds)))

    if d.get("log1p_dist_nearest_hospital_km") is None:
        dh = d.get("dist_nearest_hospital_km") or 57.0
        d["log1p_dist_nearest_hospital_km"] = float(np.log1p(max(0, dh)))

    if d.get("coastal_risk_score") is None:
        d["coastal_risk_score"] = float(
            np.exp(-d["dist_coast_km"] / 10.0) * np.exp(-d["elevation_mean"] / 30.0)
        )

    if d.get("evacuation_sufficiency") is None:
        cap = d.get("shelter_capacity_nearby") or 0
        pop = max(d.get("population") or 1, 1)
        d["evacuation_sufficiency"] = float(min(cap / pop, 10.0))

    if d.get("vulnerability_exposure") is None:
        cvi = d.get("cvi_integrated") or d.get("cvi_physical") or 3.0
        d["vulnerability_exposure"] = float(
            0.5 * (cvi / 5.0)
            + 0.3 * (1 - min(d["dist_coast_km"] / 100.0, 1.0))
            + 0.2 * d["vulnerable_pop_share"]
        )

    if d.get("infra_composite") is None:
        d["infra_composite"] = float(
            (
                d["road_connectivity_score"] / 3.0
                + d["drinking_water_score"] / 2.0
                + float(d["bus_connectivity"])
            )
            / 3.0
        )

    if d.get("monitoring_coverage") is None:
        d["monitoring_coverage"] = float(min(d["n_gauges_20km"] / 50.0, 1.0) * 0.4 + 0.6)

    row = {f: d.get(f, 0.0) for f in MODEL_FEATURES}
    return pd.DataFrame([row])


def interpret(score: float, tier: str) -> str:
    if tier == "Low":
        return (
            f"Score {score:.1f}/100 — HIGH RISK. Village has critical gaps in "
            f"shelter capacity, healthcare access, or evacuation infrastructure. "
            f"Priority intervention required before cyclone season."
        )
    elif tier == "Medium":
        return (
            f"Score {score:.1f}/100 — MODERATE RISK. Village has basic preparedness "
            f"infrastructure but significant gaps remain. Targeted improvements recommended."
        )
    else:
        return (
            f"Score {score:.1f}/100 — LOW RISK. Village demonstrates adequate preparedness "
            f"across shelter, medical, infrastructure, and early warning dimensions."
        )


def risk_label(tier: str) -> str:
    return {"Low": "HIGH", "Medium": "MODERATE", "High": "LOW"}.get(tier, "UNKNOWN")


def predict_single(v: VillageFeatures) -> PredictionResponse:
    if state.reg_pipe is None or state.cls_pipe is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_and_register.py first.")

    X = engineer_features(v)

    # Available features only
    avail = [f for f in MODEL_FEATURES if f in X.columns]
    X_in = X[avail].fillna(0)

    score = float(np.clip(state.reg_pipe.predict(X_in)[0], 0, 100))
    tier_enc = state.cls_pipe.predict(X_in)[0]
    tier = state.le.inverse_transform([tier_enc])[0]

    proba_dict = None
    if hasattr(state.cls_pipe.named_steps.get("model", object()), "predict_proba"):
        proba = state.cls_pipe.predict_proba(X_in)[0]
        proba_dict = {str(state.le.classes_[i]): round(float(p), 4) for i, p in enumerate(proba)}

    champ_reg = state.manifest.get("champion_regression", {})
    champ_cls = state.manifest.get("champion_classification", {})

    return PredictionResponse(
        preparedness_score=round(score, 2),
        preparedness_tier=tier,
        tier_probabilities=proba_dict,
        risk_level=risk_label(tier),
        interpretation=interpret(score, tier),
        model_info={
            "regressor": champ_reg.get("model_name", "unknown"),
            "classifier": champ_cls.get("model_name", "unknown"),
            "split": champ_reg.get("split_label", "unknown"),
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok" if state.reg_pipe else "degraded",
        "models_loaded": state.reg_pipe is not None,
        "startup_time": state.startup_time,
        "version": "2.0.0",
    }


@app.get("/model-info", tags=["System"])
def model_info():
    if state.manifest is None:
        raise HTTPException(status_code=503, detail="Manifest not loaded.")
    return {
        "champion_regression": state.manifest.get("champion_regression"),
        "champion_classification": state.manifest.get("champion_classification"),
        "quality_gate_thresholds": state.manifest.get("quality_gate_thresholds"),
        "registry_generated_at": state.manifest.get("generated_at"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(village: VillageFeatures):
    """
    Predict cyclone preparedness score and tier for a single village.

    Provide the raw village features; engineered features (coastal_risk_score,
    evacuation_sufficiency, etc.) are computed server-side if omitted.
    """
    return predict_single(village)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(request: BatchRequest):
    """Batch inference for multiple villages in a single request."""
    results = [predict_single(v) for v in request.villages]
    tiers = [r.preparedness_tier for r in results]
    return BatchResponse(
        count=len(results),
        predictions=results,
        summary={
            "mean_score": round(np.mean([r.preparedness_score for r in results]), 2),
            "tier_counts": {t: tiers.count(t) for t in ["Low", "Medium", "High"]},
            "high_risk_count": tiers.count("Low"),
        },
    )


@app.get("/predict/sample", tags=["Inference"])
def predict_sample():
    """Returns a sample prediction using a typical high-risk coastal village profile."""
    sample = VillageFeatures(
        dist_coast_km=1.8,
        elevation_mean=5.0,
        avg_household_size=4.8,
        sc_share=0.22,
        st_share=0.08,
        vulnerable_pop_share=0.30,
        road_connectivity_score=1.0,
        drinking_water_score=1.0,
        has_pucca_road=1,
        has_major_district_road=0,
        bus_connectivity=0,
        n_hc_10km=5,
        n_phc_chc_10km=1,
        n_gauges_20km=12,
        n_shelters_10km=0,
        cvi_physical=4.5,
        cvi_social=4.0,
        population=850,
        dist_nearest_shelter_km=8.5,
        dist_nearest_hospital_km=65.0,
        shelter_capacity_nearby=0,
        cvi_integrated=4.3,
    )
    return predict_single(sample)

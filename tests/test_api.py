"""
tests/test_api.py
=================
Phase 2 – FastAPI endpoint tests.
Tests endpoints directly and via AsyncClient / TestClient.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path before importing application modules
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import (  # noqa: E402
    BatchRequest,
    VillageFeatures,
    _load_champion_models,
    health,
    model_info,
    predict,
    predict_batch,
    predict_sample,
)

# Ensure models are loaded in memory for testing
_load_champion_models()

SAMPLE_VILLAGE = {
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


class TestHealth:
    def test_health_returns_ok(self):
        res = health()
        assert res["status"] == "ok"
        assert res["models_loaded"] is True
        assert res["version"] == "2.0.0"


class TestModelInfo:
    def test_model_info_returns_champions(self):
        res = model_info()
        assert "champion_regression" in res
        assert "champion_classification" in res
        assert res["champion_regression"]["model_name"] == "ridge"
        assert res["champion_classification"]["model_name"] == "logistic_reg"


class TestPredict:
    def test_predict_returns_valid_response(self):
        village = VillageFeatures(**SAMPLE_VILLAGE)
        res = predict(village)
        assert 0 <= res.preparedness_score <= 100
        assert res.preparedness_tier in ["Low", "Medium", "High"]
        assert res.risk_level in ["LOW", "MODERATE", "HIGH"]
        assert len(res.interpretation) > 0

    def test_predict_minimal_input(self):
        minimal = VillageFeatures(dist_coast_km=5.0, elevation_mean=20.0)
        res = predict(minimal)
        assert 0 <= res.preparedness_score <= 100
        assert res.preparedness_tier in ["Low", "Medium", "High"]

    def test_predict_extreme_coastal(self):
        extreme = dict(SAMPLE_VILLAGE)
        extreme["dist_coast_km"] = 0.1
        extreme["elevation_mean"] = 1.0
        extreme["n_shelters_10km"] = 0
        extreme["n_hc_10km"] = 1
        res = predict(VillageFeatures(**extreme))
        assert 0 <= res.preparedness_score <= 100


class TestBatchPredict:
    def test_batch_returns_summary(self):
        v1 = VillageFeatures(**SAMPLE_VILLAGE)
        v2 = VillageFeatures(**SAMPLE_VILLAGE)
        req = BatchRequest(villages=[v1, v2])
        res = predict_batch(req)
        assert res.count == 2
        assert len(res.predictions) == 2
        assert "mean_score" in res.summary
        assert "tier_counts" in res.summary


class TestSampleEndpoint:
    def test_sample_prediction(self):
        res = predict_sample()
        assert 0 <= res.preparedness_score <= 100
        assert res.risk_level in ["HIGH", "MODERATE"]


class TestInputValidation:
    def test_negative_distance_rejected(self):
        bad = dict(SAMPLE_VILLAGE)
        bad["dist_coast_km"] = -5.0
        with pytest.raises(Exception):
            VillageFeatures(**bad)

    def test_sc_share_above_1_rejected(self):
        bad = dict(SAMPLE_VILLAGE)
        bad["sc_share"] = 1.5
        with pytest.raises(Exception):
            VillageFeatures(**bad)

    def test_road_score_above_max_rejected(self):
        bad = dict(SAMPLE_VILLAGE)
        bad["road_connectivity_score"] = 5.0
        with pytest.raises(Exception):
            VillageFeatures(**bad)

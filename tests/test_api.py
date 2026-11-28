from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "distance_km": 1500.0,
    "weight_kg": 12.5,
    "hub_backlog": 1200,
    "pin_code_tier": 2,
    "declared_value_rs": 8500.0,
    "transfers": 2,
    "service": "standard",
    "destination": "Tier-2",
}


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_predict_valid():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    assert "delivery_days" in resp.json()


def test_predict_rejects_distance_over_4500():
    bad = {**VALID_PAYLOAD, "distance_km": 4501.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_negative_weight():
    bad = {**VALID_PAYLOAD, "weight_kg": -1.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422

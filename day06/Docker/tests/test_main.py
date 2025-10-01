import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# --- Corriger l'import pour trouver main.py ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

# --- 1. Test des endpoints simples ---

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()

# --- 2. Test prédiction avec un modèle non valide ---
def test_predict_invalid_model():
    response = client.post("/predict", json={"model": "invalid_model", "data": [1, 2, 3]})
    assert response.status_code == 400
    assert "error" in response.json()["detail"]

# --- 3. Test prédiction avec un modèle valide (mock) ---

@patch("main.load_model")
def test_predict_valid_model(mock_load_model):
    # Simuler le modèle chargé qui a une méthode predict
    class DummyModel:
        def predict(self, X):
            return [42]  # Valeur simulée

    mock_load_model.return_value = DummyModel()

    response = client.post("/predict", json={"model": "mock_model", "data": [1, 2, 3]})
    assert response.status_code == 200
    assert response.json() == {"prediction": [42]}

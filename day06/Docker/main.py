from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# --- Dummy load_model (sera mocké dans les tests) ---
def load_model(model_name: str):
    """Charge un modèle par son nom (ici version factice)."""
    if model_name == "iris_model":
        class DummyModel:
            def predict(self, X):
                return [sum(X)]  # exemple basique
        return DummyModel()
    else:
        raise ValueError("Unknown model")

# --- Pydantic model pour les requêtes ---
class PredictRequest(BaseModel):
    model: str
    data: list

# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI ML API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    # Exemple : liste fictive
    return {"models": ["iris_model", "mock_model"]}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        model = load_model(request.model)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid model name"})

    prediction = model.predict(request.data)
    return {"prediction": prediction}

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI(title="Credit Decisioning API")

# 🛡️ Enable CORS for Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define the expected Input Schema (Matching Lovable keys)
class CustomerRequest(BaseModel):
    age: int
    employment_status: str
    household_dependents: int
    marital_status: str
    city: str
    debt_to_income_ratio: float
    spend_to_income: float
    outstanding_liabilities: float
    # Add any other columns used in your X features

# 3. Load Artifacts once at startup
MODEL_PATH = "credit_risk_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

def load_artifacts():
    if all(os.path.exists(f) for f in [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, preprocessor, label_encoder
    return None, None, None

model, preprocessor, label_encoder = load_artifacts()

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: CustomerRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found.")

    try:
        # --- THE GATEKEEPER: HARD ELIGIBILITY RULES ---
        rejection_reason = None

        # Rule 1: Age Check (22 - 65)
        if request.age < 22 or request.age > 65:
            rejection_reason = f"Age {request.age} is outside the eligible range (22-65 years)."

        # Rule 2: Employment Check (No Retired)
        # Note: Ensure the string 'Pensioner' matches your training data exactly
        elif request.employment_status.strip().title() == "Retired":
            rejection_reason = "Retired personnel are currently not eligible for this loan product."

        # Rule 3: Debt to Income Ratio Check (> 3.0)
        elif request.debt_to_income_ratio > 3.0:
            rejection_reason = f"Debt-to-Income ratio ({request.debt_to_income_ratio}) exceeds the limit of 3.0."

        # If any rule was triggered, stop here and return the rejection
        if rejection_reason:
            return {
                "risk_label": "Ineligible",
                "reason": rejection_reason,
                "status": "Rejected",
                "confidence": 1.0
            }

        
        # STEP 1: Input from Customer
        input_data = pd.DataFrame([request.dict()])

        # STEP 2: Preprocessing (One-Hot Encoding)
        # We use the preprocessor saved in Train.py to ensure the columns match
        X_processed = preprocessor.transform(input_data)
        
        # Convert back to DataFrame if your model expects feature names (optional but safer)
        # Note: If preprocessor returns a sparse matrix, convert to dense
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        # STEP 3: Model Prediction
        # Get the label (e.g., 0, 1, 2)
        prediction_encoded = model.predict(X_processed)
        # Get the human-readable label (e.g., 'Low', 'High')
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get probabilities
        probabilities = model.predict_proba(X_processed)[0]
        max_prob = float(max(probabilities))

        # STEP 4: Credit Decisioning (Business Logic)
        # Example: Hard decline if liabilities are too high, regardless of ML
        final_decision = prediction_label
        if request.outstanding_liabilities > 5000000:
            final_decision = "Very High"

        return {
            "risk_label": final_decision,
            "confidence": round(max_prob, 2),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

import os
import joblib
import pandas as pd
import numpy as np
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
    monthly_income: float
    credit_history_type: str
    Total_Debits: float
    Total_Credits: float
    outstanding_liabilities: float
    loan_amount: float
    loan_purpose: str
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
        # STEP 1: Input from Customer
        input_data = pd.DataFrame([request.dict()])
        
        # STEP 2: RATIO CALCULATIONS  
        # Standardizing Income: Use Monthly Income if availale
        # We use .fillna(0) to avoid math errors with empty cells
        df['yearly_income'] = df ['monthly_income']*12

        # Debt to Income Ratio (DTI)
        # Calculation: Total Liabilities / Yearly Income
        # Adding 1 to denominator to prevent DivisionByZero errors
        df['debt_to_income_ratio'] = df['outstanding_liabilities'] / df['yearly_income'] 

        # Spend to Income Ratio
        # Calculation: Total Debit over 6 months / Total Income over 6 months
        df['spend_to_income'] = df['Total_Debits'] / (df['Total_Credits'])
        
        # --- THE GATEKEEPER: HARD ELIGIBILITY RULES ---
        rejection_reason = None

        # Rule 1: Age Check (22 - 65)
        if request.age < 22 or request.age > 65:
            rejection_reason = f"Age {request.age} is outside the eligible range (22-65 years)."

        # Rule 2: Employment Check (No Pensioner)
        # Note: Ensure the string 'Pensioner' matches your training data exactly
        elif request.employment_status.strip().title() == "Pensioner":
            rejection_reason = "Retired personnel are currently not eligible for this loan product."
        
        # Rule 3: Debt to Income * Spend to Income Ratio Check
        elif request.debt_to_income_ratio > 3.0:
            rejection_reason = f"Debt-to-Income ratio ({request.debt_to_income_ratio}) exceeds the limit of 3.0."
        elif request.spend_to_income > 5.0:
            rejection_reason = f"Spend-to-Income ratio ({request.spend_to_income}) exceeds the limit of 5.0."

        # If any rule was triggered, stop here and return the rejection
        if rejection_reason:
            return {
                "risk_label": "Ineligible",
                "reason": rejection_reason,
                "status": "Rejected",
                "confidence": 1.0
            }
        
        # 2. LIFE STABILITY SCORING FUNCTIONS
        def age_score(age):
            if age < 22: return 0.1
            elif age <= 25: return 0.4
            elif age <= 30 : return 0.7
            elif age <= 35: return 1.0
            elif age <= 55: return 0.6
            else: return 0.5
            
        def dependent_score(n):
            if n <= 1: return 1.0
            elif n <= 3: return 0.7
            elif n <= 5: return 0.5
            else: return 0.3
            
        def city_score(city):
            tier1 = ['Karachi', 'Lahore', 'Islamabad']
            tier2 = ['Faisalabad', 'Multan', 'Peshawar']
            if city in tier1: return 1.0
            elif city in tier2: return 0.8
            else: return 0.4
            
        def instability_penalty(row):
            penalty = 0
            if row['age'] < 30 and row['household_dependents'] >= 3:
                penalty += 0.10
            if row['employment_status'] in ['Self-Employed', 'Pensioner'] and row['household_dependents'] >= 4:
                penalty += 0.10
            if row['age'] > 55 and row['employment_status'] not in ['Salaried', 'Pensioner']:
                penalty += 0.05
            return penalty
            
        def squash(x, midpoint=0.75, steepness=6):
            return 1 / (1 + np.exp(-steepness * (x - midpoint)))
            
        input_data["yearly_income"] = input_data["monthly_income"] * 12
        input_data["half_yearly_income"] = input_data["monthly_income"] * 6
        
        # 3. APPLY LIFE STABILITY SCORING
        employment_map = {'Salaried': 1.0, 'Pensioner': 0.5, 'Self-Employed': 0.7}
        base_score = (
            0.20 * input_data['age'].apply(age_score) +
            0.30 * input_data['employment_status'].map(employment_map).fillna(0.5) +
            0.20 * input_data['household_dependents'].apply(dependent_score) +
            0.10 * input_data['marital_status'].map({'Married': 1.0, 'Single': 0.8}).fillna(0.8) +
            0.20 * input_data['city'].apply(city_score)
        )

        input_data['base_risk_score']= base_score
        
        # IMPORTANT: Add the column to the DataFrame so the model can see it
        input_data['life_stability_score'] = (base_score - input_data.apply(instability_penalty, axis=1)).clip(0, 1)
            
        # Normalization logic
        input_data['life_stability_score_adj'] = squash(input_data['life_stability_score'])
        
        # --- CRITICAL FIX: MIN-MAX NORMALIZATION ---
        # Since input_data has only 1 row, we use the global min/max observed in training.
        # Replace 0.0 and 1.0 with the actual min/max values from your training data if known.
        train_min = 0.0 
        train_max = 1.0
        input_data['life_stability_score_adj'] = (input_data['life_stability_score_adj'] - train_min) / (train_max - train_min)

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
        
        # Get probabilitiy of default
        probabilities = model.predict_proba(X_processed)[0]
        max_prob = float(max(probabilities))
        
        # PD = Prob(High) + Prob(Very High)
        pd_value = float(probabilities[0] + probabilities[3])

        # STEP 4: Credit Decisioning (Business Logic)
        # Example: Hard decline if liabilities are too high, regardless of ML
        final_decision = prediction_label
        if request.outstanding_liabilities > 5000000:
            final_decision = "Very High"

        return {
            "risk_label": final_decision,
            "probability_of_default": round(pd_value, 4),
            "confidence": round(max_prob, 2),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

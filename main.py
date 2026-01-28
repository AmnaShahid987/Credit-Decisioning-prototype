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
    yearly_income:float
    credit_history_type: str
    Total_Debits: float
    Total_Credits: float
    outstanding_liabilities: float
    loan_amount: float
    loan_purpose: str
    
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
        

        # Debt to Income Ratio (DTI)
        # Calculation: Total Liabilities / Yearly Income
        input_data['debt_to_income_ratio'] = input_data['outstanding_liabilities'] / input_data['yearly_income'] + 1

        # Spend to Income Ratio
        # Calculation: Total Debit over 6 months / Total Income over 6 months
        input_data['spend_to_income'] = input_data['Total_Debits'] / input_data['Total_Credits'] + 1
        
        # --- THE GATEKEEPER: HARD ELIGIBILITY RULES ---
        rejection_reason = None

        # Rule 1: Age Check (22 - 65)
        if request.age < 22 or request.age > 65:
            rejection_reason = f"Age {request.age} is outside the eligible range (22-65 years)."

        # Rule 2: Employment Check (No Pensioner)
        elif request.employment_status.strip().title() == "Pensioner":
            rejection_reason = "Retired personnel are currently not eligible for this loan product."
            
        # Rule 3: Outstanding Liabilities Check
        elif request.outstanding_liabilities >= 5000000:
            rejection_reason = "Your outstanding liabilities are greater than 5000000, your request cannot be processed."
        
        # Rule 4: Debt to Income * Spend to Income Ratio Check
        elif input_data['debt_to_income_ratio'].iloc[0] > 3.0:
            rejection_reason = f"Debt-to-Income ratio ({input_data['debt_to_income_ratio'].iloc[0]:.2f}) exceeds the limit of 3.0."
        elif input_data['spend_to_income'].iloc[0] > 5.0:
            rejection_reason = f"Spend-to-Income ratio ({input_data['spend_to_income'].iloc[0]:.2f}) exceeds the limit of 5.0."

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
            elif age <= 30: return 0.7
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
        
        # 3. APPLY LIFE STABILITY SCORING
        employment_map = {'Salaried': 1.0, 'Pensioner': 0.5, 'Self-Employed': 0.7}
        life_stability_score = (
            0.20 * input_data['age'].apply(age_score) +
            0.30 * input_data['employment_status'].map(employment_map).fillna(0.5) +
            0.20 * input_data['household_dependents'].apply(dependent_score) +
            0.10 * input_data['marital_status'].map({'Married': 1.0, 'Single': 0.8}).fillna(0.8) +
            0.20 * input_data['city'].apply(city_score)
        )
        
        # Apply instability penalty
        input_data['life_stability_score'] = (life_stability_score - input_data.apply(instability_penalty, axis=1)).clip(0, 1)
            
        # Normalization logic
        input_data['life_stability_score_adj'] = squash(input_data['life_stability_score'])
        
        # --- MIN-MAX NORMALIZATION ---
        train_min = 0.0 
        train_max = 1.0
        input_data['life_stability_score_adj'] = (input_data['life_stability_score_adj'] - train_min) / (train_max - train_min)

        # Calculate base_risk_score (same as training)
        input_data['base_risk_score'] = (
            0.40 * input_data['debt_to_income_ratio'].clip(0, 5) + 
            0.35 * input_data['spend_to_income'].clip(0, 2) + 
            0.25 * (1 - input_data['life_stability_score_adj'])
        )

        # STEP 2: Preprocessing (One-Hot Encoding)
        # Drop the columns that aren't features (only if they exist)
        cols_to_exclude = ['yearly_income','loan_amount', 'loan_purpose']
        X_features = input_data.drop(columns=cols_to_exclude, errors='ignore')
        
        # Use the preprocessor saved in Train.py
        X_processed = preprocessor.transform(X_features)
        
        # Convert back to DataFrame if needed
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        # STEP 3: Model Prediction
        prediction_encoded = model.predict(X_processed)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get probability of default
        probabilities = model.predict_proba(X_processed)[0]
        max_prob = float(max(probabilities))
        
        # Label encoder classes from training: ['High', 'Low', 'Medium', 'Very High']
        # Get the class order from label_encoder
        classes = label_encoder.classes_
        
        # Calculate PD = P(High) + P(Very High)
        pd_value = 0.0
        for idx, cls in enumerate(classes):
            if cls in ['High', 'Very High']:
                pd_value += float(probabilities[idx])

        # STEP 4: Credit Decisioning (Business Logic)
        def final_decision(risk_label, credit_history):
            if risk_label == 'Very High':
                return 'Decline'
            if risk_label == 'High' and credit_history == 'Thin File':
                return 'Review'
            if risk_label == 'High' and credit_history == 'Thick File':
                return 'Review'
            if risk_label == 'High' and credit_history == 'No Credit History':
                return 'Approve'
            if risk_label == 'Medium' and credit_history == 'No Credit History':
                return 'Approve'
            if risk_label == 'Medium' and credit_history == 'Thin File':
                return 'Approve'
            if risk_label == 'Medium' and credit_history == 'Thick File':
                return 'Review'
            # Default for Low risk and any remaining cases
            return 'Approve'
        
        decision = final_decsision (final_risk_label, request.credit_history_type)
        
        # Calculate base score (example formula - adjust as needed)
        base_score = int(300 + (700 * (1 - pd_value)))

        return {
            "Risk": prediction_label,
            "Credit Score": base_score,
            "Probability_of_Default": round(pd_value, 4),
            "Decision": decision,
            "confidence": round(max_prob, 2),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

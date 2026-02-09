import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI(title="Credit Decisioning API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Input Schema
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

# 3. Load models (mmap_mode saves RAM)
try:
    classification_model = joblib.load("classification_model.pkl", mmap_mode='r')
    preprocessor = joblib.load("preprocessor.pkl")
    risk_label_mapping = joblib.load("risk_label_mapping.pkl")
    # Regression model is optional since it's too large for GitHub
    regression_model = joblib.load("regression_model.pkl", mmap_mode='r') if os.path.exists("regression_model.pkl") else None
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    classification_model = preprocessor = risk_label_mapping = regression_model = None

# 4. Helper Logic (No Psychometric Questions)
def age_score(age):
    if age < 22: return 0.1
    elif age <= 25: return 0.4
    elif age <= 30: return 0.7
    elif age <= 35: return 1.0
    elif age <= 55: return 0.6
    else: return 0.5

def dependent_score(n):
    if n == 0: return 1.0
    elif n <= 2: return 0.7
    elif n <= 4: return 0.5
    else: return 0.3

def city_score(city):
    t1 = ['Karachi', 'Lahore', 'Islamabad']
    return 1.0 if city in t1 else 0.4

def instability_penalty(row):
    penalty = 0
    if row['age'] < 30 and row['household_dependents'] >= 3: penalty += 0.10
    if row['employment_status'] in ['Self-Employed', 'Pensioner'] and row['household_dependents'] >= 4: penalty += 0.10
    return penalty

def squash(x):
    return 1 / (1 + np.exp(-6 * (x - 0.75)))

@app.get("/")
def health_check():
    return {"status": "online", "classification_loaded": classification_model is not None}

@app.post("/predict")
def predict(request: CustomerRequest):
    if classification_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Models not loaded on server.")
    
    try:
        # Convert request to DataFrame immediately to fix 'df' not defined errors
        processed_df = pd.DataFrame([request.dict()])
        
        # --- Feature Engineering ---
        processed_df['yearly_income'] = processed_df['monthly_income'] * 12
        processed_df['debt_to_income_ratio'] = processed_df['outstanding_liabilities'] / (processed_df['yearly_income'] + 1)
        processed_df['new_debt_to_income_ratio'] = (processed_df['outstanding_liabilities'] + processed_df['loan_amount']) / (processed_df['yearly_income'] + 1)
        processed_df['spend_to_income'] = processed_df['Total_Debits'] / (processed_df['Total_Credits'] + 1)
        
        emp_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}
        life_base = (
            0.30 * processed_df['age'].apply(age_score) +
            0.40 * processed_df['employment_status'].map(emp_map).fillna(0.5) +
            0.20 * processed_df['household_dependents'].apply(dependent_score) +
            0.10 * processed_df['city'].apply(city_score)
        )
        
        processed_df['life_stability_score'] = (life_base - processed_df.apply(instability_penalty, axis=1)).clip(0, 1)
        processed_df['life_stability_score_adj'] = squash(processed_df['life_stability_score'])

        # --- GATEKEEPER RULES ---
        if request.age < 22 or request.age > 65:
            return {"status": "Rejected", "Decision": "Ineligible", "reason": "Age outside range."}
        if processed_df['debt_to_income_ratio'].iloc[0] > 10.0:
            return {"status": "Rejected", "Decision": "Ineligible", "reason": "High Debt-to-Income."}

        # --- ML INFERENCE ---
        excluded = ['customer_id', 'yearly_income', 'life_stability_score','loan_amount', 'loan_purpose']
        X = processed_df.drop(columns=[col for col in excluded if col in processed_df.columns])
        X_preprocessed = preprocessor.transform(X)
        
        # Classification & Probabilities
        predicted_risk_label_num = int(classification_model.predict(X_preprocessed)[0])
        class_probabilities = classification_model.predict_proba(X_preprocessed)[0]
        
        # Probability of Default (High + Very High)
        class_labels = classification_model.classes_
        prob_of_default = 0.0
        for idx, label in enumerate(class_labels):
            if label in [2, 3]: # 2=High, 3=Very High
                prob_of_default += class_probabilities[idx]
        
        risk_label_mapping_inv = {v: k for k, v in risk_label_mapping.items()}
        predicted_risk_label = risk_label_mapping_inv.get(predicted_risk_label_num, "High")
        
        # --- Decision Logic ---
        new_dti = processed_df['new_debt_to_income_ratio'].iloc[0]
        if predicted_risk_label == 'Very High' or (predicted_risk_label == 'High' and new_dti >= 1.0):
            decision = 'Decline'
        elif predicted_risk_label == 'Medium' and new_dti >= 1.5:
            decision = 'Decline'
        elif predicted_risk_label == 'High' or (predicted_risk_label == 'Medium' and new_dti >= 0.75):
            decision = 'Review'
        else:
            decision = 'Approve'

        # Score calculation (Derived from Probability of Default if Regression is missing)
        if regression_model:
            predicted_base_risk_score = float(regression_model.predict(X_preprocessed)[0])
        else:
            predicted_base_risk_score = round(float(100 * (1 - prob_of_default)), 2)

        base_score = int(300 + 550 * (1 - prob_of_default))
        max_prob = float(np.max(class_probabilities))
        
        return {
            "status": "Success",
            "Risk": predicted_risk_label,
            "Credit_Score": base_score,
            "Probability_of_Default": round(float(prob_of_default), 4),
            "Decision": decision,
            "confidence": round(max_prob, 2),
            "base_risk_score": predicted_base_risk_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

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

# 3. Load trained models and preprocessor
try:
    regression_model = joblib.load("regression_model.pkl")
    classification_model = joblib.load("classification_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    risk_label_mapping = joblib.load("risk_label_mapping.pkl")
    print("✓ Models loaded successfully")
    
except Exception as e:
    print(f"Error loading models: {e}")
    regression_model = None
    classification_model = None
    preprocessor = None
    risk_label_mapping = None


# 5. API Endpoints
@app.get("/")
def health_check():
    models_loaded = (regression_model is not None and 
                    classification_model is not None and 
                    preprocessor is not None)
    return {
        "status": "online",
        "message": "Credit Risk Assessment API",
        "models_loaded": models_loaded,
        "endpoints": {
            "/predict": "POST - Get credit risk prediction",
            "/health": "GET - Check API health"
        }
    }



@app.post("/predict")
def predict(request: CustomerRequest):
    # Check if models are loaded
    if regression_model is None or classification_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found.")
    
    try:
        # Convert request to dictionary
        input_data = request.dict()
        
        # 4. Helper Functions (from feature_engineering.py)
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
        
        def apply_feature_engineering(input_data):
            """Apply feature engineering to raw input data"""
            df = pd.DataFrame([input_data])
            
            # Calculate yearly income
            input_data['yearly_income'] = input_data['monthly_income'] * 12
            
            # Debt to Income Ratio
            input_data['debt_to_income_ratio'] = input_data ['outstanding_liabilities'] / (input_data ['yearly_income'] + 1)
            input_data['new_debt_to_income_ratio'] = ((input_data ['outstanding_liabilities'] + input_data ['loan_amount']) / (input_data ['yearly_income'] + 1))
            
            # Spend to Income Ratio
            input_data['spend_to_income'] = input_data ['Total_Debits'] / (input_data ['Total_Credits'] + 1)
            
            # Life Stability Score
            employment_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}
            
            life_base_score = (
                0.30 * df['age'].apply(age_score) +
                0.40 * df['employment_status'].map(employment_map).fillna(0.5) +
                0.20 * df['household_dependents'].apply(dependent_score) +
                0.05 * df['marital_status'].map({'Married': 1.0, 'Single': 0.8}).fillna(0.8) +
                0.10 * df['city'].apply(city_score)
            )
            
            input_data['life_stability_score'] = (life_base_score - df.apply(instability_penalty, axis=1)).clip(0, 1)
            
            # Normalization
            input_data['life_stability_score_adj'] = squash(input_data['life_stability_score'])
            min_val, max_val = 0.0, 1.0
            input_data['life_stability_score_adj'] = (input_data['life_stability_score_adj'] - min_val) / (max_val - min_val + 0.0001)
            
            return df
        

        
        # --- THE GATEKEEPER: HARD ELIGIBILITY RULES ---
        rejection_reason = None

        # Rule 1: Age Check (22 - 65)
        if request.age < 22 or request.age > 65:
            rejection_reason = f"Age {request.age} is outside the eligible range (22-65 years)."

        # Rule 2: Employment Check (No Pensioner)
        elif request.employment_status.strip().title() == "Pensioner":
            rejection_reason = "Retired personnel are currently not eligible for credit products."
            
        # Rule 3: Outstanding Liabilities Check
        elif request.outstanding_liabilities >= 5000000:
            rejection_reason = f"Your outstanding liabilities are greater than PKR 5,000,000, your request cannot be processed."
        
        # Rule 4: Debt to Income Ratio Check
        elif input_data['debt_to_income_ratio'].iloc[0] > 10.0:
            rejection_reason = f" Your outstanding debt is ({processed_df['debt_to_income_ratio'].iloc[0]:.2f}) times your income.Please clear your outstanding debt before applying again."
            
        elif input_data['new_debt_to_income_ratio'].iloc[0] > 10.0: 
            rejection_reason = f" Your loan cannot be approved because your outstanding debt will be ({processed_df['debt_to_income_ratio'].iloc[0]:.2f}) times your income.Please clear your previous outstanding debt before applying again."

        
        # Rule 5: Spend to Income Ratio Check
        elif processed_df['spend_to_income'].iloc[0] > 10.0:
            rejection_reason = f" Your transaction history shows that your debit transaction volume is ({processed_df['spend_to_income'].iloc[0]:.2f}) times greater than your income. Your request cannot be preocessed."

        # If any rule was triggered, stop here and return the rejection
        if rejection_reason:
            return {
                "status": "Rejected",
                "Decision": "Ineligible",
                "reason": rejection_reason
            }

        # --- PASS GATEKEEPER: Now run ML models ---
        
        # Prepare features for model (exclude columns not used in training)
        excluded_columns = ['customer_id', 'yearly_income', 'life_stability_score','loan_amount', 'loan_purpose']
        
        
        X = processed_df.drop(columns=[col for col in excluded_columns if col in processed_df.columns])
        
        # Preprocess features
        X_preprocessed = preprocessor.transform(X)
        
        # Make predictions
        predicted_base_risk_score = float(regression_model.predict(X_preprocessed)[0])
        predicted_risk_label_num = int(classification_model.predict(X_preprocessed)[0])
        class_probabilities = classification_model.predict_proba(X_preprocessed)[0]
        
        # Calculate probability of default (High + Very High)
        class_labels = classification_model.classes_
        prob_of_default = 0.0
        
        if 2 in class_labels:  # High
            high_idx = np.where(class_labels == 2)[0][0]
            prob_of_default += class_probabilities[high_idx]
        
        if 3 in class_labels:  # Very High
            very_high_idx = np.where(class_labels == 3)[0][0]
            prob_of_default += class_probabilities[very_high_idx]
        
        # Convert numerical label back to text
        risk_label_mapping_inv = {v: k for k, v in risk_label_mapping.items()}
        predicted_risk_label = risk_label_mapping_inv[predicted_risk_label_num]
        
        # STEP 4: Credit Decisioning (Business Logic)
        
        def final_decision(predicted_risk_label, new_debt_to_income):
            
            if risk_label == 'Very High':
                return 'Decline'
            if risk_label == 'High' and input_data['debt_to_income'] >= 2.0':
                return 'Decline'
            if risk_label == 'High' and input_data ['new_debt_to_income'] >= 1.0 :
                return 'Review'
            if risk_label == 'High' and input_data ['new_debt_to_income'] <=0.5 :
                return 'Approve'
            if risk_label == 'Medium' and input_data ['new_debt_to_income'] >=1.5:
                return 'Decline'
            if risk_label == 'Medium' and input_data ['new_debt_to_income'] >=0.75:
                return 'Review'
            if risk_label == 'Medium' and input_data ['new_debt_to_income'] <=0.75:
                return 'Approve'
            # Default for Low risk and any remaining cases
        return 'Approve'
    
        decision = final_decision(predicted_risk_label, request.new_debt_to_income)

        
        # Calculate credit score (300-850 scale based on probability of default)
        # Lower PD = Higher Score
        base_score = int(300 + 550 * (1 - prob_of_default))
        
        # Get max probability for confidence
        max_prob = float(max(class_probabilities))
        
        # Prepare response
        response = {
            "status": "Success",
            "Risk": predicted_risk_label,
            "Credit_Score": base_score,
            "Probability_of_Default": round(prob_of_default, 4),
            "Decision": decision,
            "confidence": round(max_prob, 2),
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

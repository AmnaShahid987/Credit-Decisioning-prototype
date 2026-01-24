import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so Lovable can talk to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. MATCH THE LOVABLE INPUTS
# These field names must exactly match the 'formData' keys in your React code
class CreditRequest(BaseModel):
    employmentStatus: str
    monthlyIncome: float
    maritalStatus: str
    debitTransactions: float
    creditTransactions: float
    creditHistoryType: str
    outstandingLiabilities: float
    loanAmount: float
    loanPurpose: str
    age: int
    annualIncome: float
    employmentLength: int
    creditHistoryLength: int

# 2. LOAD YOUR SPECIFIC MODEL
MODEL_NAME = "credit_risk_model.pkl"

if os.path.exists(MODEL_NAME):
    with open(MODEL_NAME, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"⚠️ Error: {MODEL_NAME} not found!")

@app.post("/predict")
def predict(data: CreditRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found on server.")

    try:
        # Convert the Pydantic object to a dictionary
        input_dict = data.dict()
        
        # Convert to DataFrame (Scikit-learn models usually expect this)
        input_df = pd.DataFrame([input_dict])
        
        # 3. RUN PREDICTION
        # Ensure your 'Feature Engineering.py' logic is either 
        # inside this function or already applied to the model pipeline
        prediction = model.predict(input_df)
        
        # Calculate probability if your model supports it
        try:
            prob = model.predict_proba(input_df)[:, 1][0]
        except:
            prob = 1.0 if prediction[0] == 1 else 0.0

        return {
            "status": "success",
            "decision": "Approved" if prediction[0] == 1 else "Denied",
            "score": int(prob * 1000), # Simulating a credit score out of 1000
            "confidence": round(float(prob), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

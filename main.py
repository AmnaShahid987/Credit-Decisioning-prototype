import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI()

# 🛡️ Allow Lovable to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Model once at startup (saves time!)
# Make sure the filename matches what Train.py produces
MODEL_PATH = "model.pkl" 

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print("⚠️ WARNING: model.pkl not found. API will start but predictions will fail.")

# 3. Define the Input Schema (Matches your Lovable form)
class CreditRequest(BaseModel):
    income: float
    age: int
    loan_amount: float
    # Add all other features your model expects

@app.post("/predict")
def predict_credit(data: CreditRequest):
    if model is None:
        return {"error": "Model not loaded on server"}

    # Convert incoming JSON to a DataFrame (Scikit-learn prefers this)
    input_df = pd.DataFrame([data.dict()])
    
    # 4. Make the Decision
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1] # Optional: confidence score

    return {
        "decision": "Approved" if prediction[0] == 1 else "Denied",
        "confidence": float(probability[0])
    }

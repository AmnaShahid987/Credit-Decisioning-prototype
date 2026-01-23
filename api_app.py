from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# IMPORT CREDIT DECISION LOGIC
from decisioning.credit_decision_logic import credit_decision

# Load trained ML model
model = joblib.load("credit_model.pkl")

@app.post("/score")
def score(application: dict):

    df = pd.DataFrame([application])

    # ML outputs
    risk_score = model.predict_proba(df)[:, 1][0]
    risk_label = model.predict(df)[0]

    # BUSINESS DECISION
    decision = credit_decision(
        risk_score=risk_score,
        risk_label=risk_label
    )

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_label": risk_label,
        "credit_decision": decision
    }



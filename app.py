from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("credit_risk_model.pkl")
pipeline = joblib.load("feature_pipeline.pkl")

@app.post("/score")
def score_application(application: dict):

    df = pd.DataFrame([application])
    features = pipeline.transform(df)

    risk_score = model.predict_proba(features)[:,1][0]
    risk_label = model.predict(features)[0]

    decision = credit_decision(risk_label, risk_score)

    return {
        "risk_score": float(risk_score),
        "risk_label": risk_label,
        "decision": decision
    }

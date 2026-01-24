import pandas as pd
import numpy as np
import joblib

# 1. LOAD DATA FIRST (Fixes the NameError)
try:
    df = pd.read_csv('train_data_final(1)(3)(1).csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Ensure the filename matches exactly in GitHub.")

# 2. DEFINE SCORING FUNCTIONS
def age_score(age):
    if age < 22: return 0.1
    elif age <= 25: return 0.4
    elif age <= 35: return 0.7
    elif age <= 55: return 1.0
    else: return 0.6

def dependent_score(n):
    if n == 0: return 0.7
    elif n <= 2: return 1.0
    elif n <= 4: return 0.8
    else: return 0.6

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

# 3. CALCULATE SCORES
employment_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}

base_score = (
    0.20 * df['age'].apply(age_score) +
    0.30 * df['employment_status'].map(employment_map) +
    0.20 * df['household_dependents'].apply(dependent_score) +
    0.10 * df['marital_status'].map({'Married': 1.0, 'Single': 0.8}) +
    0.20 * df['city'].apply(city_score)
)

df['life_stability_score'] = (base_score - df.apply(instability_penalty, axis=1)).clip(0, 1)

# Adding some randomness/noise as per your original code
df['life_stability_score'] += np.random.normal(0, 0.05, len(df))
df['life_stability_score'] = df['life_stability_score'].clip(0, 1)

# Normalizing adjusted score
df['life_stability_score_adj'] = squash(df['life_stability_score'])
min_val, max_val = df['life_stability_score_adj'].min(), df['life_stability_score_adj'].max()
df['life_stability_score_adj'] = (df['life_stability_score_adj'] - min_val) / (max_val - min_val)

# 4. FINAL RISK CALCULATION
df['base_risk_score'] = (
    0.45 * df['debt_to_income_ratio'] + 
    0.35 * df['spend_to_income'] + 
    0.20 * df['life_stability_score_adj']
)

def final_risk_label(score):
    if score > 1.0: return 'Very High'
    elif score > 0.7: return 'High'
    elif score > 0.4: return 'Medium'
    else: return 'Low'

df['final_risk_label'] = df['base_risk_score'].apply(final_risk_label)

# 5. SAVE THE RESULT FOR TRAIN.PY
# Instead of dumping a non-existent pipeline, save the processed data
df.to_csv('feature_processed_data.csv', index=False)
print("Feature Engineering complete. feature_processed_data.csv created.")

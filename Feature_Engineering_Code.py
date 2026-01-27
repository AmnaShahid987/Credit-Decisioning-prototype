
#Feature Engineering Code#
import pandas as pd
import numpy as np
import joblib

# 1. LOAD DATA
try:
    # Ensure this filename matches your GitHub file exactly
    df = pd.read_csv('Final_Dataset_modified(1).csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Check the filename in your repository.")
    exit()

# 2. RATIO CALCULATIONS
# Standardizing Income: Use Monthly Income if available, otherwise 1/6th of Half-Yearly
# We use .fillna(0) to avoid math errors with empty cells

df['yearly_income'] = df ['monthly_income']*12

# Debt to Income Ratio (DTI)
# Calculation: Total Liabilities / Monthly Income
# Adding 1 to denominator to prevent DivisionByZero errors
df['debt_to_income_ratio'] = df['outstanding_liabilities'] / df['yearly_income'] 

# Spend to Income Ratio
# Calculation: Total Debit over 6 months / Total Income over 6 months
df['spend_to_income'] = df['Total_Debits'] / (df['Total_Credits'])

# 3. SCORING FUNCTIONS
def age_score(age):
    if age < 22: return 0.1
    elif age <= 25: return 0.4
    elif age <= 30 : return 0.7
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

# 4. APPLY LIFE STABILITY SCORING
employment_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}

base_score = (
    0.20 * df['age'].apply(age_score) +
    0.30 * df['employment_status'].map(employment_map).fillna(0.5) +
    0.20 * df['household_dependents'].apply(dependent_score) +
    0.10 * df['marital_status'].map({'Married': 1.0, 'Single': 0.8}).fillna(0.8) +
    0.20 * df['city'].apply(city_score)
)

df['life_stability_score'] = (base_score - df.apply(instability_penalty, axis=1)).clip(0, 1)

# Normalization
df['life_stability_score_adj'] = squash(df['life_stability_score'])
min_val, max_val = df['life_stability_score_adj'].min(), df['life_stability_score_adj'].max()
df['life_stability_score_adj'] = (df['life_stability_score_adj'] - min_val) / (max_val - min_val)

# 5. FINAL RISK MODELING (THE TEACHER)
# This creates the target label for the ML model to learn
df['base_risk_score'] = (
    0.40 * df['debt_to_income_ratio'].clip(0, 5) + 
    0.35 * df['spend_to_income'].clip(0, 2) + 
    0.25 * (1- df['life_stability_score_adj']) # Lower stability = higher risk
)

def final_risk_label(score):
    if score > 1.0: return 'Very High'
    elif score > 0.8: return 'High'
    elif score > 0.4: return 'Medium'
    else: return 'Low'

df['final_risk_label'] = df['base_risk_score'].apply(final_risk_label)

# 6. Probability of Default# 
# Calculate min and max for base_risk_score to normalize
min_base_risk_score = df['base_risk_score'].min()
max_base_risk_score = df['base_risk_score'].max()

# Normalize base_risk_score to get probability of default (0 to 1)
# A simple min-max scaling is used here.
df['probability_of_default'] = (df['base_risk_score'] - min_base_risk_score) / (max_base_risk_score - min_base_risk_score)


# 7. Save the processed data to a CSV file
df.to_csv('feature_processed_data.csv', index=False)
print("Feature Enginering Complete. Base Risk Score, Risk Label and Probability of Default added to feature_processed_data.csv.")

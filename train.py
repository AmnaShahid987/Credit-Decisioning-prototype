import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

#Feature Engineering Code#
import pandas as pd
import numpy as np
import joblib
import os

# 1. LOAD DATA
try:
    # Ensure this filename matches your GitHub file exactly
    df = pd.read_csv('feature_processed_data.csv')
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
df['debt_to_income_ratio'] = df['outstanding_liabilities'] / df['yearly_income'] + 1

# Spend to Income Ratio
# Calculation: Total Debit over 6 months / Total Income over 6 months
df['spend_to_income'] = df['Total_Debits'] / (df['Total_Credits']) + 1 

# 3. SCORING FUNCTIONS
def age_score(age):
    if age < 22: return 0.4
    elif age <= 25: return 0.6
    elif age <= 30 : return 0.7
    elif age <= 55: return 1.0 
    elif age > 55: return 0.6
    else: return 0

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
    else: return 0.2

def instability_penalty(row):
    penalty = 0
    if row['age'] < 30 and row['household_dependents'] >= 3:
        penalty += 0.1
    if row['employment_status'] in ['Self-Employed', 'Pensioner'] and row['household_dependents'] >= 4:
        penalty += 0.1
    if row['age'] > 55 and row['employment_status'] not in ['Salaried', 'Pensioner']:
        penalty += 0.05
    return penalty

def squash(x, midpoint=0.75, steepness=6):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

# 4. APPLY LIFE STABILITY SCORING
employment_map = {'Salaried': 10, 'Pensioner': 5, 'Self-Employed': 7}

base_score = (
    0.30 * df['age'].apply(age_score) +
    0.40 * df['employment_status'].map(employment_map).fillna(0.5) +
    0.20 * df['household_dependents'].apply(dependent_score) +
    0.05 * df['marital_status'].map({'Married': 10, 'Single': 8}).fillna(0.8) +
    0.10 * df['city'].apply(city_score)
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
    if score >= 2.4: return 'Very High'
    elif score >= 1.5: return 'High'
    elif score >= 1.0: return 'Medium'
    else: return 'Low'

df['final_risk_label'] = df['base_risk_score'].apply(final_risk_label)

# 7. Save the processesd data to a CSV file 

output_filename = 'feature_processed_data.csv'

try:
    # Check if file is open elsewhere
    if os.path.exists(output_filename):
        os.remove(output_filename) # Try to delete it first to test permissions
    
    df.to_csv(output_filename, index=False)
    print(f"✅ Success! File saved as {output_filename}")

except PermissionError:
    print(f"❌ Error: Please close '{output_filename}' in Excel or other programs and try again.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

# ML MODEL TRAINING CODE#
# 1. Load the data created by Feature_Engineering.py
df = pd.read_csv('feature_processed_data.csv')

# 2. Identify Target and Features
# We drop the target and any IDs. 
# ALSO drop any 'prob_' columns if they were left over from previous runs
cols_to_drop = ['customer_id','yearly_income','final_risk_label','probability_of_default','base_risk_score'] + [c for c in df.columns if c.startswith('prob_')]
y = df['final_risk_label']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 3. Preprocess Features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

X_preprocessed = preprocessor.fit_transform(X)

# 4. Encode Target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(label_encoder.classes_)

# 5. Train Model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_preprocessed, y_encoded)
print("Model training complete.")

# 6. SAVE EVERYTHING FOR RENDER
joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Deployment artifacts saved successfully.")

import pandas as pd
import numpy as np
import os


# Define your file path
DATA_PATH = "training_feature_processed_data.csv"

def prepare_environment(file_path):
    if os.path.exists(file_path):
        print(f"Found existing data at {file_path}. Deleting to prevent skew...")
        os.remove(file_path)
    else:
        print("No existing data found. Starting fresh.")

# Execute the cleanup
prepare_environment(DATA_PATH)

# 1. LOAD DATA
try:
    df = pd.read_csv('raw_training_data.csv', index_col=False))
    print("✓ Data loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print("Error: CSV file not found. Check the filename in your repository.")
    exit()

# 2. RATIO CALCULATIONS
df['yearly_income'] = df['monthly_income'] * 12

# Debt to Income Ratio (DTI)
df['debt_to_income_ratio'] = df['outstanding_liabilities'] / (df['yearly_income'] + 1)

# Spend to Income Ratio 
df['spend_to_income'] = df['Total_Debits'] / (df['Total_Credits'] + 1)

# 3. SCORING FUNCTIONS
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

# 4. APPLY LIFE STABILITY SCORING
employment_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}

base_score = (
    0.30 * df['age'].apply(age_score) +
    0.40 * df['employment_status'].map(employment_map).fillna(0.5) +
    0.20 * df['household_dependents'].apply(dependent_score) +
    0.05 * df['marital_status'].map({'Married': 1.0, 'Single': 0.8}).fillna(0.8) +
    0.10 * df['city'].apply(city_score)
)

df['life_stability_score'] = (base_score - df.apply(instability_penalty, axis=1)).clip(0, 1)

# Normalization
df['life_stability_score_adj'] = squash(df['life_stability_score'])
min_val, max_val = df['life_stability_score_adj'].min(), df['life_stability_score_adj'].max()
df['life_stability_score_adj'] = (df['life_stability_score_adj'] - min_val) / (max_val - min_val)

# 5. FINAL RISK MODELING
df['base_risk_score'] = (
    0.40 * df['debt_to_income_ratio'].clip(0, 5) + 
    0.35 * df['spend_to_income'].clip(0, 2) + 
    0.25 * (1 - df['life_stability_score_adj'])
)

def final_risk_label(score):
    if score >= 2.4: return 'Very High'
    elif score >= 1.5: return 'High'
    elif score >= 1.0: return 'Medium'
    else: return 'Low'

df['final_risk_label'] = df['base_risk_score'].apply(final_risk_label)

print("\n--- Risk Label Distribution ---")
print(df['final_risk_label'].value_counts())

# 6. Save the processed data
output_filename = 'training_feature_processed_data.csv'

try:
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Success! File saved as {output_filename}")

except PermissionError:
    print(f"✗ Error: Please close '{output_filename}' and try again.")
except Exception as e:
    print(f"✗ An unexpected error occurred: {e}")

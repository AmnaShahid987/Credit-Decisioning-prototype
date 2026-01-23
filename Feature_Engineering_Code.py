


#Calculate life stability score for each customer

#Age score
def age_score(age):
    if age < 22:
        return 0.1
    elif age <= 25:
        return 0.4
    elif age <= 35:
        return 0.7
    elif age <= 55:
        return 1.0
    else:
        return 0.6

#Employment stability score
employment_map = {
    'Salaried': 1.0,
    'Pensioner': 0.9,
    'Self-Employed': 0.5
    }
#Marital Status score

marital_status = {
    'Single': 0.8,
    'Married': 1.0
    }

def dependent_score(n):
    if n == 0:
        return 0.7
    elif n <= 2:
        return 1.0
    elif n <= 4:
        return 0.8
    else:
        return 0.6

tier1 = ['Karachi', 'Lahore', 'Islamabad']
tier2 = ['Faisalabad', 'Multan', 'Peshawar']

def city_score(city):
    if city in tier1:
        return 1.0
    elif city in tier2:
        return 0.8
    else:
        return 0.4

#lifestabilityscore
base_score = (
    0.20 * df['age'].apply(age_score) +
    0.30 * df['employment_status'].map(employment_map) +
    0.20 * df['household_dependents'].apply(dependent_score) + \
    0.10 * df['marital_status'].map({'Married':1.0, 'Single':0.8}) +
    0.20 * df['city'].apply(city_score)
)

#penalizeinstabilityinteractions

def instability_penalty(row):
    penalty = 0

    # Young + many dependents
    if row['age'] < 30 and row['household_dependents'] >= 3:
        penalty += 0.10

    # Gig/self-employed + high dependents
    if row['employment_status'] in ['Self-Employed', 'Pensioner'] and row['household_dependents'] >= 4:
        penalty += 0.10

    # Older + no pension
    if row['age'] > 55 and row['employment_status'] not in ['Salaried', 'Pensioner']:
        penalty += 0.05

    return penalty

df['life_stability_score'] = (
    base_score - df.apply(instability_penalty, axis=1)
).clip(0, 1)

import numpy as np

df['life_stability_score'] += np.random.normal(0, 0.05, len(df))
df['life_stability_score'] = df['life_stability_score'].clip(0, 1)

df['life_stability_score'].describe()

import numpy as np

def squash(x, midpoint=0.75, steepness=6):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

df['life_stability_score_adj'] = squash(df['life_stability_score'])

min_val = df['life_stability_score_adj'].min()
max_val = df['life_stability_score_adj'].max()

df['life_stability_score_adj'] = (
    (df['life_stability_score_adj'] - min_val) / (max_val - min_val)
)

#HARD ELIGIBILITY CRITERIA

# Define eligibility criteria
age_eligible_condition = (df['age'] >= 22) & (df['age'] <= 65)
liabilities_eligible_condition = (df['outstanding_liabilities'] <= 5000000) # Assuming 5,000,0000 was a typo and meant 5,000,000

# Combine conditions to find eligible customers
eligible_customers = df[age_eligible_condition & liabilities_eligible_condition]

# Find ineligible customers (for reporting purposes)
ineligible_customers = df[~(age_eligible_condition & liabilities_eligible_condition)]

print(f"Number of eligible customers: {len(eligible_customers)}")
print(f"Number of ineligible customers: {len(ineligible_customers)}")

print("\nFirst five eligible customers:")
display(eligible_customers.head())

# Define the mapping for credit_history_type to numerical values
credit_history_mapping = {
    'No Credit History': 0,
    'Thin File': 1,
    'Thick File': 2
}

#Basic Risk Score Model

df['base_risk_score'] = (
    0.45 * df['debt_to_income_ratio'] + \
    0.35 * df['spend_to_income'] + \
    0.20 * df['life_stability_score_adj']
)

#Assigning risk label
def final_risk_label(score):
    if score > 1.0:
        return 'Very High'
    elif score > 0.7:
        return 'High'
    elif score > 0.4:
        return 'Medium'
    else:
        return 'Low'
df['final_risk_label'] = df['base_risk_score'].apply(final_risk_label)


import joblib

joblib.dump(feature_pipeline, "feature_pipeline.pkl")

import pandas as pd
from sklearn.model_selection import train_test_split

# Define the features you want to use for X
# IMPORTANT: Customize this list with your actual feature column names from model_df
X_features = [
    'monthly_income',
    'debt_to_income_ratio',
    'spend_to_income',
    'age',
    'household_dependents',
    'life_stability_score_adj',
    'stability_adj',
    'credit_history_adj',
    'base_risk_score'
    # Add or remove features as needed
]

# Ensure all selected features exist in model_df
# This step checks for missing columns and informs you if any are not found.
missing_features = [f for f in X_features if f not in model_df.columns]
if missing_features:
    print(f"Warning: The following features were not found in 'model_df' and will be excluded: {missing_features}")
    X_features = [f for f in X_features if f in model_df.columns]
    if not X_features:
        raise ValueError("No valid features remaining after filtering. Please check your X_features list.")

# Select the specified features for X
X = model_df[X_features]

# Define the target variable y
y = model_df['final_risk_label']

# Split the data into training (80%) and testing (20%) sets
# random_state ensures reproducibility
# stratify=y ensures that the proportion of target labels is the same in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set (X_train) size: {len(X_train)} rows")
print(f"Test set (X_test) size: {len(X_test)} rows")
print(f"Training set (y_train) size: {len(y_train)} rows")
print(f"Test set (y_test) size: {len(y_test)} rows")

display(X_train.head())
display(y_train.head())

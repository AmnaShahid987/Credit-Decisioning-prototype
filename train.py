import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


## Task
#Load the `training_feature_processed_data.csv` dataset, prepare features and targets (regression target: `base_risk_score`, classification target: `final_risk_label`, 
#excluding `customer_id`, `yearly_income`, `life_stability_score`, `base_risk_score`, `final_risk_label` from features), apply one-hot encoding to categorical features, 
#train a regression model for `base_risk_score` and a classification model for `final_risk_label`, generate predictions including `probability_of_default` on the training data, 
#and finally, confirm successful training of both models by displaying a sample of the generated `base_risk_score`, `final_risk_label`, and `probability_of_default` values.


# 1. Load the processed data created by Feature_Engineering.py

df = pd.read_csv('training_feature_processed_data.csv')
print('Dataset loaded successfully. Shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())

# 2. Prepare Features and Target variables

### Subtask:
#Identify and separate the features (X) and target variables (y) from the loaded training data. `y_regression` will be `base_risk_score`, and `y_classification` will be `final_risk_label`. 
#The specified columns (`customer_id`, `yearly_income`, `life_stability_score`, `base_risk_score`, `final_risk_label`) will be explicitly excluded from the feature set (X). 
#Convert `final_risk_label` to numerical ordinal values for classification.

# A. Define the list of columns to be excluded from the feature set (X)
excluded_columns = [
    'customer_id',
    'yearly_income',
    'life_stability_score',
    'base_risk_score',
    'final_risk_label'
]

# B. Create the feature DataFrame X by dropping these excluded columns from df
X = df.drop(columns=excluded_columns)

# C. Create the regression target Series y_regression from the base_risk_score column of df
y_base_risk_score = df['base_risk_score']

# D. Create the classification target Series y_classification from the final_risk_label column of df
y_final_risk_label = df['final_risk_label']

# E. Map the categorical values in y_classification ('Low', 'Medium', 'High', 'Very High')
# to numerical ordinal values (e.g., 0, 1, 2, 3 respectively) and store them back in y_classification.
risk_label_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very High': 3
}
y_final_risk_label_encoded = y_final_risk_label.map(risk_label_mapping)

# Display the unique values of y_classification and their counts to verify the conversion.
print('\n--- Data Shapes ---')
print('Features (X) shape:', X.shape)
print('Regression Target shape:', y_base_risk_score.shape)
print('Classification Target shape:', y_final_risk_label_encoded.shape)
print('\nRisk Label Distribution:')
print(y_final_risk_label_encoded.value_counts().sort_index())


# 3. Preprocess Features (One-Hot Encoding)
#The next step is to apply one-hot encoding to the categorical features within the feature DataFrame X. 

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'\nCategorical columns to encode: {categorical_cols}')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

X_preprocessed = preprocessor.fit_transform(X)
print(f'Shape after preprocessing: {X_preprocessed.shape}')

# 4. Train Regression Model (for base_risk_score)
print('\n--- Training Regression Model ---')
regression_model = RandomForestRegressor(random_state=42, n_estimators=100)
regression_model.fit(X_preprocessed, y_base_risk_score)
print('✓ Regression model trained successfully')

# 5. Train Classification Model (for final_risk_label)
print('\n--- Training Classification Model ---')
classification_model = RandomForestClassifier(random_state=42, n_estimators=100)
classification_model.fit(X_preprocessed, y_final_risk_label_encoded)
print('✓ Classification model trained successfully')

# 6. Generate Predictions (on training data for verification)
print('\n--- Generating Predictions ---')
predicted_base_risk_score = regression_model.predict(X_preprocessed)
predicted_final_risk_label = classification_model.predict(X_preprocessed)
class_probabilities = classification_model.predict_proba(X_preprocessed)

# Calculate probability of default (High + Very High risk)
class_labels = classification_model.classes_
prob_of_default = np.zeros(len(X_preprocessed))

if 2 in class_labels:  # High
    high_idx = np.where(class_labels == 2)[0][0]
    prob_of_default += class_probabilities[:, high_idx]

if 3 in class_labels:  # Very High
    very_high_idx = np.where(class_labels == 3)[0][0]
    prob_of_default += class_probabilities[:, very_high_idx]

print('\nSample Predictions (first 5):')
print('Base Risk Score:', predicted_base_risk_score[:5])
print('Final Risk Label:', predicted_final_risk_label[:5])
print('Probability of Default:', prob_of_default[:5])

# 7. Save Models and Preprocessor
print('\n--- Saving Models ---')
joblib.dump(regression_model, "regression_model.pkl")
joblib.dump(classification_model, "classification_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(risk_label_mapping, "risk_label_mapping.pkl")
print('✓ All models saved successfully!')
print('\nSaved files:')
print('  - regression_model.pkl')
print('  - classification_model.pkl')
print('  - preprocessor.pkl')
print('  - risk_label_mapping.pkl')

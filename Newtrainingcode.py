Train.py code 



import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

#Load Training Data# 
df = pd.read_csv('training_feature_processed_data.csv')
print('Dataset loaded successfully. First 5 rows:')
print(df.head())


#Prepare Features and Targets 

# 1. Define the list of columns to be excluded from the feature set (X)
excluded_columns = [
    'customer_id',
    'yearly_income',
    'life_stability_score',
    'base_risk_score',
    'final_risk_label'
]

# 2. Create the feature DataFrame X by dropping these excluded columns from df
X = df.drop(columns=excluded_columns)

# 3. Create the regression target Series y_regression from the base_risk_score column of df
base_risk_score_regression = df['base_risk_score']

# 4. Create the classification target Series y_classification from the final_risk_label column of df
final_risk_label_classification = df['final_risk_label']

# 5. Map the categorical values in final_risk_label_classification ('Low', 'Medium', 'High', 'Very High')
# to numerical ordinal values (e.g., 0, 1, 2, 3 respectively) and store them back in final_risk_label_classification.
risk_label_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very High': 3
}
final_risk_label_classification = final_risk_label_classification.map(risk_label_mapping)


#Preprocessing

# Identify categorical columns in X
categorical_cols = X.select_dtypes(include='object').columns

# Apply one-hot encoding to categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print('Shape of X after one-hot encoding:', X_encoded.shape)
print('\nFirst 5 rows of X_encoded:')
print(X_encoded.head())


#Train regression model for predicting base_risk_score 


# Initialize and train the regression model
regression_model = RandomForestRegressor(random_state=42)
regression_model.fit(X_encoded, y_regression)

print('Regression model (RandomForestRegressor) trained successfully.')


#Train risk classification model 

# Initialize and train the classification model
classification_model = RandomForestClassifier(random_state=42)
classification_model.fit(X_encoded, y_classification)

print('Classification model (RandomForestClassifier) trained successfully.')

import numpy as np

# Generate regression predictions for base_risk_score
predicted_base_risk_score = regression_model.predict(X_encoded)

# Generate classification predictions for final_risk_label
predicted_final_risk_label = classification_model.predict(X_encoded)

# Generate probability of default (probability of 'High' or 'Very High' risk classes)
# Map numerical labels back to original for understanding
# 'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3
# Probability of default could be sum of probabilities for 'High' (2) and 'Very High' (3)

class_probabilities = classification_model.predict_proba(X_encoded)

# Assuming risk_label_mapping is available from previous steps
# If not, define it again:
risk_label_mapping_inverse = {
    0: 'Low',
    1: 'Medium',
    2: 'High',
    3: 'Very High'
}

# Identify column indices for 'High' and 'Very High'
# The order of classes in predict_proba is classification_model.classes_
# Let's verify and map
class_labels = classification_model.classes_
prob_of_high_risk = np.zeros(len(X_encoded))
prob_of_very_high_risk = np.zeros(len(X_encoded))

if 2 in class_labels:
    high_risk_idx = np.where(class_labels == 2)[0][0]
    prob_of_high_risk = class_probabilities[:, high_risk_idx]

if 3 in class_labels:
    very_high_risk_idx = np.where(class_labels == 3)[0][0]
    prob_of_very_high_risk = class_probabilities[:, very_high_risk_idx]

probability_of_default = prob_of_high_risk + prob_of_very_high_risk


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


# 1. Load the data created by Feature_Engineering.py

df = pd.read_csv('training_feature_processed_data.csv')
print('Dataset loaded successfully. First 5 rows:')
print(df.head())

## Prepare Features and Target variables

### Subtask:
#Identify and separate the features (X) and target variables (y) from the loaded training data. `y_regression` will be `base_risk_score`, and `y_classification` will be `final_risk_label`. 
#The specified columns (`customer_id`, `yearly_income`, `life_stability_score`, `base_risk_score`, `final_risk_label`) will be explicitly excluded from the feature set (X). 
#Convert `final_risk_label` to numerical ordinal values for classification.

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
y_base_risk_score = df['base_risk_score']

# 4. Create the classification target Series y_classification from the final_risk_label column of df
y_final_risk_label = df['final_risk_label']

# 5. Map the categorical values in y_classification ('Low', 'Medium', 'High', 'Very High')
# to numerical ordinal values (e.g., 0, 1, 2, 3 respectively) and store them back in y_classification.
risk_label_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very High': 3
}
y_final_risk_label = y_final_risk_label.map(risk_label_mapping)

# Display the unique values of y_classification and their counts to verify the conversion.
print('Features (X) shape:', X.shape)
print('Regression Target (y_regression) shape:', y_regression.shape)
print('Classification Target (y_classification) shape:', y_classification.shape)
print('\nUnique values and counts for y_classification (numerical):')
print(y_classification.value_counts())
print('\nFirst 5 rows of X:')
print(X.head())
print('\nFirst 5 values of y_regression:')
print(y_regression.head())
print('\nFirst 5 values of y_classification:')
print(y_classification.head())

#The next step is to apply one-hot encoding to the categorical features within the feature DataFrame X. 
#I will identify the categorical columns and use pd.get_dummies to transform them, dropping the first category to avoid multicollinearity.

# Identify categorical columns in X
categorical_cols = X.select_dtypes(include='object').columns

# Apply one-hot encoding to categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print('Shape of X after one-hot encoding:', X_encoded.shape)
print('\nFirst 5 rows of X_encoded:')
print(X_encoded.head())


################## 3. Preprocess Features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

X_preprocessed = preprocessor.fit_transform(X)

############### 4. Encode Target
label_encoder = LabelEncoder()
base_risk_score_encoded = label_encoder.fit_transform(y)
print(label_encoder.classes_)


# Initialize and train the regression model for base risk score
base_risk_score_model = RandomForestRegressor(random_state=42)
base_risk_score_model.fit(X_encoded, y_base_risk_score)

print('Regression model for base risk score is trained successfully.')


# Initialize and train the risk classification model
final_risk_label_model = RandomForestClassifier(random_state=42)
final_risk_label_model.fit(X_encoded, y_classification)

print('Classification model for final risk labels trained successfully.')

#Now that both the regression and classification models are trained, I will generate predictions for `base_risk_score` and `final_risk_label` 
#using the trained models on the `X_encoded` training data. Additionally, I will generate `probability_of_default` from the classification model.

# Generate regression predictions for base_risk_score
predicted_base_risk_score = regression_base_risk_score.predict(X_encoded)

# Generate classification predictions for final_risk_label
predicted_final_risk_label = final_risk_label_model.predict(X_encoded)

# Generate probability of default (probability of 'High' or 'Very High' risk classes)
# Map numerical labels back to original for understanding
# 'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3
# Probability of default could be sum of probabilities for 'High' (2) and 'Very High' (3)

class_probabilities = final_risk_label_model.predict_proba(X_encoded)

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

class_labels = final_risk_label_model.classes_
prob_of_high_risk = np.zeros(len(X_encoded))
prob_of_very_high_risk = np.zeros(len(X_encoded))

if 2 in class_labels:
    high_risk_idx = np.where(class_labels == 2)[0][0]
    prob_of_high_risk = class_probabilities[:, high_risk_idx]

if 3 in class_labels:
    very_high_risk_idx = np.where(class_labels == 3)[0][0]
    prob_of_very_high_risk = class_probabilities[:, very_high_risk_idx]

probability_of_default = prob_of_high_risk + prob_of_very_high_risk

print('Predictions generated successfully.')
print('\nSample of predicted base_risk_score:', predicted_base_risk_score[:5])
print('Sample of predicted final_risk_label (numerical):', predicted_final_risk_label[:5])
print('Sample of predicted probability_of_default:', probability_of_default[:5])


# 6. SAVE EVERYTHING FOR RENDER
joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Deployment artifacts saved successfully.")

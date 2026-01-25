import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# 1. Load the data created by Feature_Engineering.py
df = pd.read_csv('feature_processed_data.csv')

# 2. Identify Target and Features
# We drop the target and any IDs. 
# ALSO drop any 'prob_' columns if they were left over from previous runs
cols_to_drop = ['customer_id', 'final_risk_label'] + [c for c in df.columns if c.startswith('prob_')]
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

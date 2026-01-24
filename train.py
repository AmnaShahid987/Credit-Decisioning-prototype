# Load the CSV file into a pandas DataFrame
df = pd.read_csv('feature_processed_data.csv')

## ML model####

# Task
The task is to build a supervised classification model to predict `final_risk_label` using the `train_data_final (1) (3) (1).csv` dataset. This involves loading the dataset, identifying relevant features and the target variable, preprocessing the data (including one-hot encoding categorical features and handling missing values), training a RandomForestClassifier model, and evaluating its performance using cross-validation (calculating accuracy, precision, recall, F1-score, and generating a confusion matrix).


## Identify Target and Features

Define the target variable as `final_risk_label` and identify appropriate features for the model. This will involve separating the DataFrame into features (X) and target (y).

y = df['final_risk_label']
X = df.drop(columns=['customer_id', 'final_risk_label'])


## Preprocess Features (X)

### Subtask:
One-hot encode categorical features in `X` to create `X_preprocessed`.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Create a preprocessor for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns as they are
)

# Apply the preprocessing to X
X_preprocessed = preprocessor.fit_transform(X)

# Convert X_preprocessed back to a DataFrame for easier inspection and consistency if needed
# Get feature names after one-hot encoding
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
other_feature_names = [col for col in X.columns if col not in categorical_cols]
all_feature_names = list(encoded_feature_names) + list(other_feature_names)

X_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names, index=X.index)


## Model Selection and Training & Model Evaluation using Cross-Validation

### Subtask:
Train a RandomForestClassifier model and evaluate its performance using cross-validation (e.g., K-fold cross-validation) on the full preprocessed dataset. This will involve calculating metrics such as accuracy, precision, recall, and F1-score across multiple folds, and generating a confusion matrix.


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 2. Initialize a LabelEncoder and fit it to the target variable y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Instantiate a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)

# 4. Set up K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Use cross_val_predict to generate predictions
y_pred_cv = cross_val_predict(model, X_preprocessed, y_encoded, cv=kf)

# 6. Calculate the overall accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_encoded, y_pred_cv)
precision = precision_score(y_encoded, y_pred_cv, average='weighted')
recall = recall_score(y_encoded, y_pred_cv, average='weighted')
f1 = f1_score(y_encoded, y_pred_cv, average='weighted')

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall Precision (weighted): {precision:.4f}")
print(f"Overall Recall (weighted): {recall:.4f}")
print(f"Overall F1-score (weighted): {f1:.4f}")

# 7. Generate and print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_encoded, y_pred_cv, target_names=label_encoder.classes_))

# 8. Generate and display a confusion matrix
conf_matrix = confusion_matrix(y_encoded, y_pred_cv)
print("\nConfusion Matrix:")
display(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

#Model Fitting
model.fit(X_preprocessed, y_encoded)

--------------------------------------------
## Probability of Default for Training Data
--------------------------------------------
import pandas as pd

# Get probability predictions for the training set using the fitted model
# X_preprocessed and label_encoder are already defined from previous steps
proba_predictions_train = model.predict_proba(X_preprocessed)

# Create a DataFrame for better visualization and merging
proba_df_train = pd.DataFrame(proba_predictions_train, columns=[f'prob_{cls}' for cls in label_encoder.classes_], index=X_preprocessed.index)

# Merge these probabilities back into the original training DataFrame (df)
# Ensure indexes align correctly
df = df.merge(proba_df_train, left_index=True, right_index=True)

---------------------------------------------------
## Save Probability of Default for Training Data
----------------------------------------------------
### Subtask:
Calculate the probability of default for the training data, add it to the original `df` DataFrame, and save the updated DataFrame to `train_data_final (1) (3) (1).csv`.
import pandas as pd

# Define the target variable as 'final_risk_label'
y = df['final_risk_label']

# Define the feature matrix X by dropping 'customer_id' and 'final_risk_label'
X = df.drop(columns=['customer_id', 'final_risk_label'])

print("DataFrame and X, y defined.")

import joblib

joblib.dump(model, "credit_risk_model.pkl")



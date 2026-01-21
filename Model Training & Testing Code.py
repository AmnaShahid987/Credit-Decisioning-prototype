# Load the CSV file into a pandas DataFrame
df = pd.read_csv('train_data_final (1) (3).csv')

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


-------------------------------------------
## Model Validation
-------------------------------------------
import pandas as pd

# Define the mapping for credit_history_type to numerical values (same as training)
credit_history_mapping = {
    'No Credit History': 0,
    'Thin File': 1,
    'Thick File': 2
}

# Create the new 'credit_history_encoded' column in df_test
df_test['credit_history_encoded'] = df_test['credit_history_type'].map(credit_history_mapping)

# Re-define X_test, ensuring base_risk_score and credit_history_encoded are included as features
X_test = df_test.drop(columns=['customer_id', 'final_risk_label']) # Now, base_risk_score is implicitly kept as a feature

# Apply the *same* preprocessor fitted on the training data to the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Convert X_test_preprocessed back to a DataFrame
X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=all_feature_names, index=X_test.index)

# Encode the actual target variable for the test set using the same label_encoder
y_test_encoded = label_encoder.transform(y_test)

print("Shape of X_test_preprocessed:", X_test_preprocessed.shape)
print("First 5 rows of X_test_preprocessed:")
display(X_test_preprocessed.head())

print("\nFirst 5 rows of encoded y_test:")
display(y_test_encoded[:5])

##Model Fitting
model.fit(X_preprocessed, y_encoded)

# Make predictions on the preprocessed test data
y_pred_test = model.predict(X_test_preprocessed)

print("First 10 predicted labels on test data:")
print(label_encoder.inverse_transform(y_pred_test[:10]))
print("\nFirst 10 actual labels on test data:")
print(label_encoder.inverse_transform(y_test_encoded[:10]))

--------------------------------------------
## Probability of Default for Test Data
--------------------------------------------

import pandas as pd

# Get probability predictions for the test set
# model was already fitted in cell `507ec958`
proba_predictions = model.predict_proba(X_test_preprocessed)

# Create a DataFrame for better visualization
proba_df = pd.DataFrame(proba_predictions, columns=label_encoder.classes_, index=X_test.index)

# Display the first few rows of the probabilities
print("First 5 rows of Probability of Default (by risk label) for Test Data:")
display(proba_df.head())

# Optionally, you can also display alongside the predicted risk label for context
print("\nFirst 5 rows with Predicted Risk Label for Test Data:")
predicted_labels_df = pd.DataFrame({
    'customer_id': df_test['customer_id'].loc[X_test.index],
    'Predicted_Risk_Label': label_encoder.inverse_transform(y_pred_test)
}, index=X_test.index)
display(pd.concat([predicted_labels_df, proba_df], axis=1).head())

--------------------------------------------
##Save Probability of Default for Test Data
--------------------------------------------
import pandas as pd

# proba_df contains the probabilities for the test set, indexed correctly.
# df_test is the original test DataFrame.

# Merge the probabilities back into the original test DataFrame (df_test)
# Ensure indexes align correctly
df_test = df_test.merge(proba_df, left_index=True, right_index=True)

# Save the modified DataFrame back to the CSV file
df_test.to_csv('test_data_final_updated.csv', index=False)

print("Modified DataFrame (df_test) with probability of default columns saved to 'test_data_final_updated.csv'.")
print("First 5 rows of the updated DataFrame with new probability columns:")
display(df_test.head())
----------------------------------------------------------
## Model Metrics
-----------------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Calculate overall metrics on the test set
accuracy_test = accuracy_score(y_test_encoded, y_pred_test)
precision_test = precision_score(y_test_encoded, y_pred_test, average='weighted')
recall_test = recall_score(y_test_encoded, y_pred_test, average='weighted')
f1_test = f1_score(y_test_encoded, y_pred_test, average='weighted')

print(f"Test Set Accuracy: {accuracy_test:.4f}")
print(f"Test Set Precision (weighted): {precision_test:.4f}")
print(f"Test Set Recall (weighted): {recall_test:.4f}")
print(f"Test Set F1-score (weighted): {f1_test:.4f}")

# Generate and print a detailed classification report for the test set
print("\nTest Set Classification Report:")
print(classification_report(y_test_encoded, y_pred_test, target_names=label_encoder.classes_))

# Generate and display a confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test_encoded, y_pred_test)
print("\nTest Set Confusion Matrix:")
display(pd.DataFrame(conf_matrix_test, index=label_encoder.classes_, columns=label_encoder.classes_))  

##Features of Misclassified 'Low' to 'Medium' Risk Sample

# Get the encoded values for 'Low' and 'Medium' risk
low_risk_encoded = label_encoder.transform(['Low'])[0]
medium_risk_encoded = label_encoder.transform(['Medium'])[0]

# Find indices where actual is 'Low' and predicted is 'Medium'
misclassified_low_to_medium_indices = (
    (y_test_encoded == low_risk_encoded) & (y_pred_test == medium_risk_encoded)
)

# Filter X_test to get the features of these misclassified samples
misclassified_features = X_test.loc[X_test.index[misclassified_low_to_medium_indices]]

print(f"Number of samples misclassified from 'Low' to 'Medium': {len(misclassified_features)}")
print("Features of the misclassified samples (first 5 rows if more than 5):")
display(misclassified_features.head())

print("Summary statistics for misclassified 'Low' to 'Medium' risk samples:")
display(misclassified_features.describe())

-------------------------------------------------------
## Apply Credit Decision Logic to Test Data
-------------------------------------------------------
import pandas as pd

# Define the credit_decision function as in the original notebook
def credit_decision(row):

    if row['final_risk_label'] == 'Very High':
       return 'Decline'
    if row['final_risk_label'] == 'High'and row['credit_history_type'] == 'Thin File':
        return 'Review'
    if row['final_risk_label'] == 'High' and row['credit_history_type'] == 'Thick File':
         return 'Review'
    if row['final_risk_label']== 'High' and row['credit_history_type'] == 'No Credit History':
        return 'Approve'
    if row['final_risk_label'] == 'Medium' and row['credit_history_type'] == 'No Credit History':
     return 'Approve'
    if row['final_risk_label'] == 'Medium' and row['credit_history_type'] == 'Thin File':
     return 'Approve'
    if row['final_risk_label'] == 'Medium' and row['credit_history_type'] == 'Thick File':
     return 'Review'
    # Default for Low risk and any remaining Medium risk cases not caught above
    return 'Approve'

# Apply the credit_decision function to df_test to create the 'decision' column
df_test['decision'] = df_test.apply(credit_decision, axis=1)

print("First 5 rows of df_test with the new 'decision' column:")
display(df_test.head())

print("\nCounts of each decision type:")
display(df_test['decision'].value_counts())

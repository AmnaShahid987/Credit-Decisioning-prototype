

-------------------------------------------------------
## Apply Credit Decision Logic to Training Data
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

# Apply the credit_decision function to df_train to create the 'decision' column
df_train['decision'] = df_train.apply(credit_decision, axis=1)

print("First 5 rows of df_train with the new 'decision' column:")
display(df_train.head())

print("\nCounts of each decision type:")
display(df_test['decision'].value_counts())

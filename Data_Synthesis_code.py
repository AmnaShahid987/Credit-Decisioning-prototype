

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000  # Seed size for SDV

# -------------------------
# 1. AGE (22–65)
# -------------------------
age = np.random.randint(22, 66, N)

# -------------------------
# 2. MARITAL STATUS
# -------------------------
marital_status = []
for a in age:
    if a < 30:
        marital_status.append(np.random.choice(["Single", "Married"], p=[0.8, 0.2]))
    elif a < 45:
        marital_status.append(np.random.choice(["Single", "Married"], p=[0.45, 0.55]))
    else:
        marital_status.append(np.random.choice(["Single", "Married"], p=[0.25, 0.75]))

# -------------------------
# 3. EMPLOYMENT STATUS
# -------------------------
employment_status = []
for a in age:
    if a < 55:
        employment_status.append(
            np.random.choice(
                ["Salaried", "Self-Employed"],
                p=[0.6, 0.4]
            )
        )
    else:
        employment_status.append(
            np.random.choice(
                ["Salaried", "Self-Employed", "Pensioner"],
                p=[0.3, 0.2, 0.5]
            )
        )

# -------------------------
# 4. CITY
# -------------------------
cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"]

city = np.random.choice(
    cities,
    size=N,
    p=[0.35, 0.30, 0.15, 0.10, 0.10]
)

# -------------------------
# 5. HOUSEHOLD DEPENDENTS (1–6)
# -------------------------
dependents = []
for ms, a in zip(marital_status, age):
    if ms == "Single":
        dependents.append(
            np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        )
    else:  # Married → dependents >= 2
        if a < 30:
            dependents.append(np.random.choice([2, 3], p=[0.6, 0.4]))
        elif a < 45:
            dependents.append(np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2]))
        else:
            dependents.append(np.random.choice([3, 4, 5, 6], p=[0.3, 0.3, 0.25, 0.15]))

# -------------------------
# Assemble DataFrame
# -------------------------
seed_customer_df = pd.DataFrame({
    "age": age,
    "marital_status": marital_status,
    "employment_status": employment_status,
    "city": city,
    "household_dependents": dependents
})

# -------------------------
# Hard constraint checks (DO NOT SKIP)
# -------------------------
assert seed_customer_df["age"].between(22, 65).all()
assert seed_customer_df["household_dependents"].between(1, 6).all()
assert (seed_customer_df.loc[
    seed_customer_df["marital_status"] == "Married",
    "household_dependents"
] >= 2).all()

seed_customer_df.head()

seed_customer_df.info()
seed_customer_df.describe(include="all")

# Married → dependents >= 2
assert (seed_customer_df.loc[
    seed_customer_df["marital_status"] == "Married",
    "household_dependents"
] >= 2).all()

import numpy as np
import pandas as pd

# Copy dataset to avoid mutating original
fixed_seed_df = seed_customer_df.copy()


# Reassign employment status
fixed_seed_df.loc[invalid_pensioners, "employment_status"] = np.random.choice(
    ["Salaried", "Self-Employed"],
    size=invalid_pensioners.sum(),
    p=[0.65, 0.35]
)

# -------------------------
# Hard validation checks
# -------------------------
assert (
    fixed_seed_df.loc[
        fixed_seed_df["employment_status"] == "Pensioner",
        "age"
    ] >= 60
).all()

# -------------------------
# Save corrected dataset
# -------------------------
fixed_seed_df.to_csv("seed_customers_corrected.csv", index=False)

fixed_seed_df.head()

#Checking for biases

def show_distribution(df, column):
    return (
        df[column]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

show_distribution(seed_customer_df, "marital_status")
show_distribution(seed_customer_df, "employment_status")
show_distribution(seed_customer_df, "city")

seed_customer_df["age"].hist(bins=20)

pd.crosstab(
    seed_customer_df["age"] // 10 * 10,
    seed_customer_df["marital_status"],
    normalize="index"
)

pd.crosstab(
    seed_customer_df["employment_status"],
    seed_customer_df["age"] // 10 * 10,
    normalize="index"
)

seed_customer_df.groupby("marital_status")["household_dependents"].describe()

pd.crosstab(
    seed_customer_df["city"],
    seed_customer_df["employment_status"],
    normalize="index"
)

seed_customer_df[["age", "household_dependents"]].corr()

#Fairness Check
seed_customer_df["eligible"] = (
    (seed_customer_df["age"] <= 60) &
    (seed_customer_df["employment_status"] != "Pensioner") &
    (seed_customer_df["household_dependents"] <= 5)
).astype(int)

pd.crosstab(
    seed_customer_df["employment_status"],
    seed_customer_df["eligible"],
    normalize="index"
)


pd.crosstab(
    fixed_seed_df["age"] // 10 * 10,
    fixed_seed_df["employment_status"],
    normalize="index"
)

import pandas as pd

# fixed_seed_df is already defined and corrected from previous cells.

# Remove existing customer_id if it exists to avoid ValueError
if 'customer_id' in fixed_seed_df.columns:
    fixed_seed_df = fixed_seed_df.drop(columns=['customer_id'])

# Add customer_id directly to fixed_seed_df
# Modify this to only include numerical values
fixed_seed_df.insert(
    0,
    "customer_id",
    [int(str(i).zfill(6)) for i in range(1, len(fixed_seed_df) + 1)]
)

fixed_seed_df.head()


# -------------------------
# Save corrected dataset
# -------------------------
# Save the updated fixed_seed_df (now with customer_id)
fixed_seed_df.to_csv("seed_customers_corrected_1.csv", index=False)

fixed_seed_df.head()

fixed_seed_df.to_csv("seed_customers_corrected_1.csv", index=False)
print("DataFrame saved successfully to 'seed_customers_corrected_1.csv'")

fixed_seed_df = fixed_seed_df.drop(columns=['eligible'])
fixed_seed_df.head()

fixed_seed_df.to_csv("seed_customers_final.csv", index=False)
print("DataFrame saved successfully to 'seed_customers_final.csv'")

"""# Task
Generate a synthetic dataset of 50,000 records using the Synthetic Data Vault (SDV) library, based on the patterns learned from the `fixed_seed_df` loaded from "seed_customers_final.csv". Save the generated synthetic data to a CSV file named "synthetic_customers.csv".
----------------------------
## Install SDV Library
----------------------------
### Subtask:
Install the Synthetic Data Vault (SDV) library, if not already installed, to enable synthetic data generation.

**Reasoning**:
The subtask is to install the `sdv` library. I will use `!pip install sdv` in a code cell to install the library.
"""



!pip install sdv
print("SDV library installed successfully.")

"""**Reasoning**:
The subtask is to generate synthetic data. I will first load the `fixed_seed_df` from 'seed_customers_final.csv', then import the `SDV` library, define the metadata, train a `CTGAN` model on the real data, generate 50,000 synthetic records, and finally save the synthetic data to 'synthetic_customers.csv'.


import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

# 1. Load the fixed_seed_df from the saved CSV file
fixed_seed_df = pd.read_csv('seed_customers_final.csv')
print("Loaded fixed_seed_df from 'seed_customers_final.csv'")

# 2. Create a metadata object directly from the DataFrame for a single table
metadata = Metadata.single_table_from_dataframe(fixed_seed_df)
print("Metadata created from 'fixed_seed_df'.")

# 3. Set 'customer_id' as the primary key
metadata.set_primary_key(column_name='customer_id')
print("Primary key 'customer_id' set.")

# 4. Save the metadata for replicability
metadata.save_to_json('synthetic_customers_metadata.json')
print("Metadata saved to 'synthetic_customers_metadata.json'")

# 5. Instantiate CTGANSynthesizer model with the metadata object
print("\nTraining CTGAN model...")
model = CTGANSynthesizer(metadata)

# 6. Fit the CTGANSynthesizer model to the fixed_seed_df
model.fit(fixed_seed_df)
print("CTGAN model trained successfully.")

# 7. Generate 50,000 synthetic records
print("\nGenerating 50,000 synthetic records...")
synthetic_data = model.sample(num_rows=50000)
print("Synthetic data generated successfully.")

# Save the generated synthetic data to a CSV file
synthetic_data.to_csv("synthetic_customers.csv", index=False)
print("\nSynthetic data saved to 'synthetic_customers.csv'.")

# 8. Display the head of the synthetic data
synthetic_data.head()

---------------------------------------------
#CAPACITY
---------------------------------------------
import numpy as np

# 1. Initialize variables
MIN_YEARLY_INCOME = 600000
MAX_YEARLY_INCOME = 6000000
NO_LIABILITY_PERCENTAGE = 0.40
LOW_LIABILITY_PERCENTAGE = 0.30
HIGH_LIABILITY_PERCENTAGE = 0.30

# 2. Define helper function for assigning income bands
def assign_income_band(income, min_income, max_income):
    mid_point = min_income + (max_income - min_income) / 2
    if income <= mid_point:
        return 'Lower'
    else:
        return 'Higher'

# 3. Define helper function for generating outstanding liabilities
def generate_outstanding_liabilities(yearly_income):
    rand_val = np.random.rand()
    if rand_val < NO_LIABILITY_PERCENTAGE:
        return 0  # 40% chance of 0 liability
    elif rand_val < (NO_LIABILITY_PERCENTAGE + LOW_LIABILITY_PERCENTAGE):
        # 30% chance of liability between yearly_income and 2 * yearly_income
        return np.random.uniform(yearly_income, 2 * yearly_income)
    else:
        # 30% chance of liability between 2 * yearly_income and 5 * yearly_income
        return np.random.uniform(2 * yearly_income, 5 * yearly_income)

# 4. Define helper function for assigning credit history type (logic to be detailed later)
def assign_credit_history_type(outstanding_liabilities, yearly_income_band):
    # This function will be implemented with specific logic later.
    # For now, it returns a placeholder.
    if outstanding_liabilities == 0:
        return 'No Credit History'
    elif yearly_income_band == 'Lower' and outstanding_liabilities > 0:
        return 'Thin File' # Example placeholder logic
    else:
        return 'Thick File' # Example placeholder logic

##Generate Income
import numpy as np

# 1. Generate an initial 'yearly_income' for all customers
MIN_YEARLY_INCOME = 600000
MAX_YEARLY_INCOME = 6000000

df_customers['yearly_income'] = np.random.uniform(
    MIN_YEARLY_INCOME, MAX_YEARLY_INCOME, size=len(df_customers))

# 2. Define a list of urban cities
urban_cities = ['Karachi', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Multan', 'Hyderabad'] # Common urban cities in Pakistan

# 3. Identify customers who meet the criteria
higher_income_criteria = (
    (df_customers['age'] >= 30) &
    (df_customers['age'] <= 45) &
    (df_customers['employment_status'] == 'Salaried') &
    (df_customers['city'].isin(urban_cities))
)

# 4. For these identified customers, update their 'yearly_income'
higher_half_min_income = (MIN_YEARLY_INCOME + MAX_YEARLY_INCOME) / 2
df_customers.loc[higher_income_criteria, 'yearly_income'] = np.random.uniform(
    higher_half_min_income, MAX_YEARLY_INCOME, size=higher_income_criteria.sum())

# 5. Display the head of the df_customers DataFrame
print("df_customers head with adjusted 'yearly_income' for target group:")
print(df_customers[['customer_id', 'age', 'employment_status', 'city', 'yearly_income']].head())

df_customers['monthly_income'] = df_customers['yearly_income'] / 12

##Generate Outstanding Liabilities
df_customers['outstanding_liabilities'] = df_customers['yearly_income'].apply(generate_outstanding_liabilities)

print("Head of df_customers with 'yearly_income' and 'outstanding_liabilities':")
print(df_customers[['yearly_income', 'outstanding_liabilities']].head())

##Generate Credit History 
df_customers['yearly_income_band'] = df_customers['yearly_income'].apply(lambda x: assign_income_band(x, MIN_YEARLY_INCOME, MAX_YEARLY_INCOME))
df_customers['credit_history_type'] = df_customers.apply(
    lambda row: assign_credit_history_type(row['outstanding_liabilities'], row['yearly_income_band']),
    axis=1
)

print("Head of df_customers with 'yearly_income', 'yearly_income_band', 'outstanding_liabilities', and 'credit_history_type':")
print(df_customers[['yearly_income', 'yearly_income_band', 'outstanding_liabilities', 'credit_history_type']].head())

print("Verifying generated columns in df_customers:")

# 1. Display the first few rows of the df_customers DataFrame with specific columns
print("\nHead of df_customers with generated columns:")
print(df_customers[['yearly_income', 'monthly_income', 'outstanding_liabilities', 'yearly_income_band', 'credit_history_type']].head())

# 2. Print descriptive statistics for 'yearly_income', 'monthly_income', and 'outstanding_liabilities'
print("\nDescriptive statistics for yearly_income, monthly_income, and outstanding_liabilities:")
print(df_customers[['yearly_income', 'monthly_income', 'outstanding_liabilities']].describe())

# 3. Print the value counts for 'yearly_income_band' and 'credit_history_type'
print("\nValue counts for 'yearly_income_band':")
print(df_customers['yearly_income_band'].value_counts())

print("\nValue counts for 'credit_history_type':")
print(df_customers['credit_history_type'].value_counts())

## Debt-To-Income Ratio Calculation
# Calculate 'debt_to_income_ratio'
# Handle potential division by zero or infinite values gracefully
# If yearly_income is 0, the ratio should be 0 or NaN, not infinity.
df_customers['debt_to_income_ratio'] = df_customers['outstanding_liabilities'] / df_customers['yearly_income']

# Replace infinite values (which can result from yearly_income being 0) with NaN
df_customers['debt_to_income_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill any NaN values (including those from division by zero) with 0
df_customers['debt_to_income_ratio'].fillna(0, inplace=True)

print("Head of df_customers with 'outstanding_liabilities', 'yearly_income', and 'debt_to_income_ratio':")
display(df_customers[['outstanding_liabilities', 'yearly_income', 'debt_to_income_ratio']].head())

print("\nDescriptive statistics for 'debt_to_income_ratio':")
display(df_customers['debt_to_income_ratio'].describe())
---------------------------------------------
##DISCIPLINE 
---------------------------------------------
# 1. Convert 'Transaction Date' to datetime objects if not already done
df_transactions_loaded['Transaction Date'] = pd.to_datetime(df_transactions_loaded['Transaction Date'])

# 2. Extract Year and Month
df_transactions_loaded['Year'] = df_transactions_loaded['Transaction Date'].dt.year
df_transactions_loaded['Month'] = df_transactions_loaded['Transaction Date'].dt.month

# 3. Define transaction types as debit or credit based on previously defined lists
# Ensure 'Salary Credit' and 'Pension Credit' are considered incoming/credit
all_credit_categories = set(credit_categories + additional_credit_types)

def get_transaction_type_group(category):
    if category in all_credit_categories:
        return 'Credit'
    elif category in debit_categories:
        return 'Debit'
    return 'Other'

df_transactions_loaded['Transaction Type Group'] = df_transactions_loaded['Transaction Category'].apply(get_transaction_type_group)

# Filter out 'Other' if any categories weren't explicitly defined
df_transactions_filtered = df_transactions_loaded[df_transactions_loaded['Transaction Type Group'].isin(['Credit', 'Debit'])]

# 4. Group by customer_id, Year, Month, and Transaction Type Group, then sum the amounts
monthly_summary = df_transactions_filtered.groupby(['customer_id', 'Year', 'Month', 'Transaction Type Group'])['Transaction Amount'].sum().reset_index()

# 5. Pivot the table to get 'Total Monthly Debits' and 'Total Monthly Credits' as separate columns
monthly_summary_pivot = monthly_summary.pivot_table(
    index=['customer_id', 'Year', 'Month'],
    columns='Transaction Type Group',
    values='Transaction Amount',
    fill_value=0
).reset_index()

# Rename columns for clarity
monthly_summary_pivot.rename(columns={
    'Credit': 'Total Monthly Credits',
    'Debit': 'Total Monthly Debits'
}, inplace=True)

print("Monthly summary of total debit and credit transactions per customer:")
display(monthly_summary_pivot.head())


# Ensure df_customers is loaded (it should be from previous steps)
# df_customers = pd.read_csv('customer_financial_profile_seed (1).csv')

# Ensure df_customer_totals_loaded is loaded (it should be from previous steps)
# df_customer_totals_loaded = pd.read_csv('customer_total_transactions.csv')

# Merge the two DataFrames on 'customer_id'
df_merged_customer_data = pd.merge(
    df_customers,
    customer_total_transactions, # Using the DataFrame already in memory
    on='customer_id',
    how='left'
)

print("Merged customer data with total debit and credit transactions:")
display(df_merged_customer_data.head())


df_merged_customer_data['Half Yearly Income'] = df_merged_customer_data['monthly_income'] * 6

print("Merged customer data with 'Half Yearly Income' added:")
display(df_merged_customer_data.head())

# Calculate 'spend to income' ratio
df_merged_customer_data['spend_to_income'] = df_merged_customer_data['Total_Debits'] / df_merged_customer_data['Total_Credits']

# Handle potential division by zero or infinite values gracefully
df_merged_customer_data['spend_to_income'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_merged_customer_data['spend_to_income'].fillna(0, inplace=True) # Replace NaN with 0 or another suitable value

print("Merged customer data with 'spend to income' added:")
display(df_merged_customer_data.head())

df_merged_customer_data.to_csv('merged_customer_data.csv', index=False)
print("Updated merged customer data (including spend to income) saved to 'merged_customer_data.csv'. You can now download this file.")

df_merged_customer_data_loaded = pd.read_csv('merged_customer_data.csv')

unique_customers_in_merged_data = df_merged_customer_data_loaded['customer_id'].nunique()

print(f"The number of unique customers in 'merged_customer_data.csv' is: {unique_customers_in_merged_data}")

df_merged_customer_data.to_csv('merged_customer_data.csv', index=False)













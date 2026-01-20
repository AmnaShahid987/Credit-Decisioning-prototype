
# Data Synthesis Summary (SDV and CTGANSynthesizer)

This document summarizes the process, rules, and logic applied to generate synthetic customer data from a seed dataset using the SDV (Synthetic Data Vault) library, specifically the CTGANSynthesizer.

## 1. Objective

The primary goal was to generate synthetic customer data that mimics the statistical properties of an original seed dataset, while enforcing specific constraints, notably that all generated customer cities must be from a predefined list of major Pakistani cities.
This synthetic data can be used for privacy-preserving data sharing, augmenting small datasets, or testing applications.

## 2. Tools Used

*   **pandas**: For data loading, manipulation, and filtering.
*   **SDV (Synthetic Data Vault)**: The core library for generating synthetic data, utilizing `Metadata` for schema definition and `CTGANSynthesizer` for model training and generation.

## 3. Data Synthesis, Preparation and Filtering
A two step process was used to generate seed data (3000 rows) , using the below constraints to reflect some of the dynamics of Pakistan.

Seed Data Generation
1. Life Stability : 
   a. Age Range: 18–65 years
   b. Gender : 70% Male, 30% Female 
   c. Marital Status with values as either Single or Married  
      Logical constraints:
      Males < 25 → Single
      Females < 20 → Single
      A greater proportion of people between 30-65 to be married 
   d. Household Dependents : Range: 0–6
      Distribution logic: i. Lower income bands → higher dependents 2. Higher income bands → fewer dependents
   e. Geographical Location : City - all major Pakistani cities with distribution skewed towards urban areas (Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad)
   f. Employment Status : Salaried, Self-Employed, Pensioner - Salaried and self-employed are skewed towards urban areas (Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad) 
2. Capacity 
   a. Yearly Income Range: PKR 600,000 – 6,000,000  
      Logical Constraint : Those in the age group 30-45, salaried and living in urban areas have a higher yearly income 
   b. Outstanding Liabilities : 
      Logical Constraint: 
        i. 0 for 40% of the population
        ii. greater than yearly income but less than twice the yearly income for 30% of the population 
        iii. greater than twice the yearly income but less than 5 times the yearly income for 30% of the population
   c. Credit History Type ( No credit history, Thin File, Thick File) 
      Logical Constraint: 
      i.  'No Credit History' for 40% of the population and must have 0 outstanding liabilities must be in the lower income band
      ii. 'Thin File' should have less than twice the outstanding liabilities and must fall in the lower income band
      iii. 'Thick File' should be skewed towards higher income bands. 

3.Discipline 
    Transactional Data of past 6 months for each customer was generated using the below constraints: 

There is a 60% chance for a non-recurring transaction to be a debit transaction, chosen randomly from debit_categories
(e.g., 'Cash Withdrawal', 'Outgoing Debit', 'Debit Card', 'Utility Bill Payments etc.).
There is a 40% chance for a non-recurring transaction to be a credit transaction, chosen randomly from credit_categories 
(e.g., 'Incoming Credit', 'Cash Deposit', 'Incoming Remittances', 'Investment Income').

1. Recurring Transactions  
   Logical Constraints : 
   i. If employment_status is 'Salaried', a 'Salary Credit' transaction is generated. 
   ii. If employment_status is 'Pensioner', a 'Pension Credit' transaction is generated. (Note: The current dataset does not have 'Retired' status, but the logic is in place for it).
   iii. Remittance : Lower income bands should have monthly incoming remittance transactions, of at least 50,000 - 100,000 and higher income bands should have less frequent and higher amount remittance transactions. 
                Frequency: These recurring credits are generated monthly for eligible customers.
                Amount: The Transaction Amount for recurring credits is set directly equal to the customer's monthly_income.
       i. For credit transactions, the Transaction Amount is a random value between 5% and 20% of the customer's monthly_income.
    v. Utility Bill Payments : At least one utitlity bill payment transaction every month for every customer. The transaction amount can vary as per below constraints : 
        Logical Constraints
        Range: PKR 10,000 – 100,000 
        Lower for higher income bands and higher for lower income bands (Increasing shift to solar panels for energy needs)


2. Non-Recurring Transactions 
   Logical Constraints :
   i.  For debit transactions, the Transaction Amount is a random value between 1% and 15% of the customer's monthly_income.i.e. Cash Withdrawal, Debit Card) 
   ii. For credit transactions, the Transaction Amount is a random value between 5% and 20% of the customer's monthly_income. i.e. Cash Deposit, Incoming Credit Transaction should be more frequent for those 'Self Employed' : 
       Monthly Debit Limit Enforcement:
      The total sum of all debit transactions for a customer within a given month is monitored. 
      A newly generated debit transaction is only added if the monthly_debit_total (sum of existing debits for that month)
      plus the transaction_amount of the new debit does not exceed 120% of the customer's monthly_income.
      This allows for a buffer, meaning customers can spend up to 20% more than their monthly income through debits. 
      If a debit transaction would push the monthly total beyond this limit, that specific debit transaction is discarded.


## 4.Synthetic Data Generation: SDV was used to generate data of 50,000 unique customers across life stability, capacity and discipline.   
    

## 5. Comparative Data Analysis

Comparative analysis through visualization (histograms, KDE plots for numerical features; grouped bar charts for categorical features) between the original and 
synthetic data showed:
*   **Numerical Features (`age`, `household_dependents`, 'yearly income','outstanding liabilities', )**: The synthetic data successfully replicated the distributions, showing similar peaks and shapes.
*   **Categorical Features (`marital_status`, `employment_status`, `city`, 'credit history type')**: The proportions and distributions of categories were closely matched, demonstrating 
that the `CTGANSynthesizer` effectively learned these patterns. 
*   **All generated cities were valid Pakistani cities.


import streamlit as st
import datetime
import requests

st.set_page_config(page_title="Credit Decisioning Model", layout="centered")

st.title("Enter details to receive a risk classificiation and a credit decision")

# -----------------------------
# Date of Birth Input
# -----------------------------
st.subheader("Applicant Details")

st.write("Birth Date")

col1, col2, col3 = st.columns(3)

with col1:
    day = st.selectbox("Day", list(range(1, 32)))

with col2:
    month = st.selectbox(
        "Month",
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    )

with col3:
    current_year = datetime.datetime.now().year
    year = st.selectbox("Year", list(range(1955, current_year + 1)))

month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

birth_date = datetime.date(year, month_map[month], day)
today = datetime.date.today()
age = today.year - birth_date.year - (
    (today.month, today.day) < (birth_date.month, birth_date)

# -----------------------------
# Employment Status
# -----------------------------
employment_status = st.selectbox(
    "Employment Status",
    ["Select employment status", "Salaried", "Self-Employed", "Pensioner"]
)

# -----------------------------
# Marital Status
# -----------------------------
marital_status = st.selectbox(
    "Marital Status",
    ["Select marital status","Single", "Married"]
)

# -----------------------------
# Household Dependents
# -----------------------------
household_dependents = st.selectbox(
    "Household Dependents",
    ["Select number of household dependents"] + list(range(1, 7))
)


# -----------------------------
# Monthly Income Slider
# -----------------------------
monthly_income = st.slider(
    "Monthly Income",
    min_value=0,
    max_value=1000000,
    step=50_000,
    format="%d"
)



#-----------------------------------------
#Outstanding Liabilities
#-----------------------------------------

outstanding_liabilities = st.slider(
    "Outstanding Liabilities (PKR)",
    min_value=0,
    max_value=5_000_000,
    value=0,
    step=50_000
)
#------------------------------------
#Total Debit and Credit of past 6 months
#-------------------------------------
total_debit_6m = st.slider(
    "Total Debit Amount (Last 6 Months) (PKR)",
    min_value=0,
    max_value=10_000_000,
    value=0,
    step=100_000
)

total_credit_6m = st.slider(
    "Total Credit Amount (Last 6 Months) (PKR)",
    min_value=0,
    max_value=10_000_000,
    value=0,
    step=100_000
)

# ---- Validation Logic ----
all_fields_filled = (
    outstanding_liabilities > 0 and
    total_debit_6m > 0 and
    total_credit_6m > 0
)

# -----------------------------
# Submit Application
# -----------------------------
if st.button("Submit"):
    if not all_fields_filled:
        st.error("All fields are mandatory. Please provide values greater than zero.")
    else:
        st.success("Financial data captured successfully.")

        # Display captured values (for prototype transparency)
        st.write("### Captured Inputs")
        st.write(f"- Outstanding Liabilities: PKR {outstanding_liabilities:,}")
        st.write(f"- Total Debit (6 months): PKR {total_debit_6m:,}")
        st.write(f"- Total Credit (6 months): PKR {total_credit_6m:,}")

            
    if employment_status == "Select employment status":
        st.error("Please select employment status")

    elif marital_status == "Select marital Status":
        st.error("Please select marital status")

    elif household_dependents == "Select number of household dependents":
        st.error("Please select number of household dependents")

    else:
        st.success("Application Submitted")
        
      # Placeholder for next step (API / model call)
        st.info("Proceeding to credit decisioning logic...")

    
    payload = {
        "age": age,
        "employment_status": employment_status,
        "marital_status": marital_status,
        "household_dependents": household_dependents,
        "monthly_income": monthly_income
        "total_debit_6m": total_debit_6m
        "total_credit_6m": total_credit_6m
    }

    st.subheader("Calculated Inputs Sent to Model")
    st.json(payload)

    # Call FastAPI backend
    try:
        response = requests.post("http://localhost:8000/score", json=payload)
        result = response.json()

        st.subheader("Credit Decision Output")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Score", round(result.get("risk_score", 0), 4))
        with col2:
            st.metric("Credit Decision", result.get("decision", "N/A"))

    except Exception as e:
        st.error("Backend service is not available. Please start the FastAPI server.")

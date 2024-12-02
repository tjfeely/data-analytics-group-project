# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
)
from sklearn.utils.validation import check_is_fitted
import streamlit as st
import joblib
import wrds

# Step 1: Connect to WRDS and Retrieve Data
db = wrds.Connection()

# Query to retrieve financial data
query = """
SELECT gvkey, tic, datadate, at, lt, ni, act, lct, bkvlps
FROM comp.funda
WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
AND datadate >= '2018-01-01'
"""
data = db.raw_sql(query)

# Step 2: Clean and Process Data
data.dropna(inplace=True)  # Drop rows with missing values

# Calculate financial ratios
data['Current_Ratio'] = data['act'] / data['lct']
data['ROA'] = data['ni'] / data['at']
data['Debt_to_Equity'] = data['lt'] / (data['at'] - data['lt'])

# Replace infinity or extremely large values
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Add a bankruptcy indicator column (BKVLPS <= 0 => Bankruptcy)
data['Bankruptcy'] = np.where(data['bkvlps'] <= 0, 1, 0)

# Prepare data for machine learning
features = ['Current_Ratio', 'ROA', 'Debt_to_Equity']
X = data[features]
y = data['Bankruptcy']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train or Load a Pre-trained Model
model_path = "random_forest_model.pkl"
try:
    model = joblib.load(model_path)
    print("Loaded pre-trained model.")
except FileNotFoundError:
    print("No pre-trained model found. Training a new model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Model trained successfully and saved.")

# Step 4: Ensure Model is Fitted
def ensure_model_is_fitted(model):
    try:
        check_is_fitted(model)
    except:
        print("The model is not fitted. Training the model now...")
        model.fit(X_train, y_train)

# Step 5: Evaluate Model
def evaluate_model():
    ensure_model_is_fitted(model)  # Ensure the model is fitted
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Bankruptcy', 'Bankruptcy'])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ROC-AUC Score
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC Curve")
    plt.show()

# Step 6: Predict Bankruptcy for a Selected Company
def predict_bankruptcy_for_company(company_identifier, use_recent=True):
    ensure_model_is_fitted(model)  # Ensure the model is fitted
    dataset = data if not use_recent else data.groupby('gvkey').last().reset_index()
    company_data = dataset[(dataset['gvkey'] == company_identifier) | (dataset['tic'] == company_identifier)]

    if company_data.empty:
        print(f"No data found for company: {company_identifier}")
        return

    company_name = company_data['tic'].iloc[0]
    gvkey = company_data['gvkey'].iloc[0]

    company_features = company_data[features]
    predicted_label = model.predict(company_features)[0]
    probability = model.predict_proba(company_features)[:, 1]

    print(f"Prediction for {company_name} ({gvkey}): {predicted_label} (probability: {probability[0]:.2f})")

# Step 7: Backtest with Historical Data for Bankrupt Companies
def backtest_bankruptcy():
    ensure_model_is_fitted(model)  # Ensure the model is fitted
    bankrupt_companies = data[data['Bankruptcy'] == 1]
    print("Bankrupt Companies in the Dataset:")
    print(bankrupt_companies[['gvkey', 'tic']].drop_duplicates())

    while True:
        company_identifier = input("Enter a gvkey or ticker (tic) to backtest, or type 'exit' to quit: ").strip()
        if company_identifier.lower() == 'exit':
            print("Exiting backtest mode.")
            break
        predict_bankruptcy_for_company(company_identifier, use_recent=False)

# Step 8: Streamlit Integration
st.title("Bankruptcy Prediction System")
st.write("This tool allows you to predict bankruptcy for companies based on financial ratios.")

if st.button("Evaluate Model"):
    evaluate_model()

input_gvkey_tic = st.text_input("Enter gvkey or ticker (tic) to predict bankruptcy:")
if st.button("Predict Bankruptcy"):
    if input_gvkey_tic:
        predict_bankruptcy_for_company(input_gvkey_tic)
    else:
        st.error("Please enter a valid gvkey or ticker (tic).")

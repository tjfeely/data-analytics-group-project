# Import necessary libraries
import wrds
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
 
# Step 1: Connect to WRDS and Retrieve Data
db = wrds.Connection()
 
# Query to retrieve financial data
query = """
SELECT gvkey, tic, datadate, at, lt, ni, act, lct, bkvlps
FROM comp.funda
WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
AND datadate >= '2010-01-01'
"""
data = db.raw_sql(query)
 
# Step 2: Clean and Process Data
data.dropna(inplace=True)  # Drop rows with missing values
 
# Calculate financial ratios
data['Current_Ratio'] = data['act'] / data['lct']
data['ROA'] = data['ni'] / data['at']
data['Debt_to_Equity'] = data['lt'] / (data['at'] - data['lt'])
 
# Replace infinity or extremely large values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
 
# Add a bankruptcy indicator column (BKVLPS <= 0 -> Bankruptcy)
data['Bankruptcy'] = np.where(data['bkvlps'] <= 0, 1, 0)
 
# Step 3: Prepare Data for Machine Learning
features = ['Current_Ratio', 'ROA', 'Debt_to_Equity']
X = data[features]
y = data['Bankruptcy']
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Step 4: Train a Machine Learning Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
 
print("Model trained successfully!")
 
# Step 5: Evaluate Model
def evaluate_model():
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
 
    # Feature Importances
    importances = model.feature_importances_
    plt.barh(features, importances)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.show()
 
# Step 6: Backtest with Historical Data for Bankrupt Companies
def backtest_bankruptcy():
    bankrupt_companies = data[data['Bankruptcy'] == 1]
    print("Bankrupt Companies in the Dataset:")
    print(bankrupt_companies[['gvkey', 'tic']].drop_duplicates())
 
    while True:
        company_identifier = input("\nEnter a gvkey or ticker (tic) to backtest, or type 'exit' to quit: ").strip()
        if company_identifier.lower() == 'exit':
            print("Exiting backtest mode.")
            break
        predict_bankruptcy_for_company(company_identifier, use_recent=False)
 
# Step 7: Predict Bankruptcy for a Selected Company
def predict_bankruptcy_for_company(company_identifier, use_recent=True):
    dataset = data if not use_recent else data.groupby('gvkey').last().reset_index()
    company_data = dataset[(dataset['gvkey'] == company_identifier) | (dataset['tic'] == company_identifier)]
   
    if company_data.empty:
        print(f"No data found for company: {company_identifier}")
        return
 
    company_name = company_data['tic'].iloc[0]
    gvkey = company_data['gvkey'].iloc[0]
 
    company_features = company_data[features]
    prediction = model.predict(company_features)
    probability = model.predict_proba(company_features)[:, 1]
 
    print(f"\nCompany Name: {company_name} (GVKEY: {gvkey})")
    print(f"Bankruptcy Prediction: {'Bankruptcy' if prediction[0] else 'No Bankruptcy'}")
    print(f"Probability of Bankruptcy: {probability[0]:.2f}")
 
# Step 8: User Interaction Loop
def main():
    while True:
        print("\nOptions:")
        print("1. Evaluate Model")
        print("2. Analyze a Specific Company")
        print("3. Backtest with Historical Data")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()
 
        if choice == '1':
            evaluate_model()
        elif choice == '2':
            company_identifier = input("\nEnter a gvkey or ticker (tic): ").strip()
            predict_bankruptcy_for_company(company_identifier)
        elif choice == '3':
            backtest_bankruptcy()
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
 
# Run the program
main()
 
# Close WRDS connection
db.close()

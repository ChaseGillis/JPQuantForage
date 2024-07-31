"""Here is your task
The risk manager has collected data on the loan borrowers. The data is in tabular format,
with each row providing details of the borrower, including their income, total loans outstanding,
and a few other metrics. There is also a column indicating if the borrower has previously
defaulted on a loan. You must use this data to build a model that, given details for any loan
described above, will predict the probability that the borrower will default (also known as PD: the
probability of default). Use the provided data to train a function that will estimate the probability of
default for a borrower. Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

You should produce a function that can take in the properties of a loan and output the expected loss.
You can explore any technique ranging from a simple regression or a decision tree to something more
advanced. You can also use multiple methods and provide a comparative analysis.
Submit your code below."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    # Check for missing values
    # print(df.isnull().sum())
    # Basic data exploration
    # print(df.describe())
    return df

def train_model(df):
    # Feature and target extraction
    X = df.drop(columns=['customer_id', 'default'])
    y = df['default']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model training (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
    
    return model, scaler

def predict_expected_loss(model, scaler, loan_details, recovery_rate=0.1):
    loan_details_scaled = scaler.transform([loan_details])
    pd = model.predict_proba(loan_details_scaled)[:, 1][0]
    loan_amt = loan_details[2]  # Assuming the third element is loan amount
    expected_loss = pd * (1 - recovery_rate) * loan_amt
    return expected_loss

def main():
    pd.set_option("display.max_columns", None)
    path = 'Task 3 and 4 Loan Data.csv'
    df = load_and_preprocess_data(path)
    model, scaler = train_model(df)
    # Example prediction
    sample_loan_details = [5, 10000, 5000, 30000, 5, 650]  # Replace with actual values
    loss = predict_expected_loss(model, scaler, sample_loan_details)
    print(f"Expected Loss: ${loss:.2f}")

if __name__ == "__main__":
    main()
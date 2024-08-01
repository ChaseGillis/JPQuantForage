"""Charlie wants to make her model work for future data sets, so she needs a general 
approach to generating the buckets. Given a set number of buckets corresponding to 
the number of input labels for the model, she would like to find out the boundaries 
that best summarize the data. You need to create a rating map that maps the FICO score 
of the borrowers to a rating where a lower rating signifies a better credit score.

The process of doing this is known as quantization. You could consider many ways of solving 
the problem by optimizing different properties of the resulting buckets, such as the mean 
squared error or log-likelihood (see below for definitions). For background on quantization,
see here.

Mean squared error

You can view this question as an approximation problem and try to map all the entries in 
a bucket to one value, minimizing the associated squared error. We are now looking to minimize 
the following: 

Log-likelihood

A more sophisticated possibility is to maximize the following log-likelihood function:

Where bi is the bucket boundaries, ni is the number of records in each bucket, ki is the number 
of defaults in each bucket, and pi = ki / ni is the probability of default in the bucket. This 
function considers how rough the discretization is and the density of defaults in each bucket. 
This problem could be addressed by splitting it into subproblems, which can be solved incrementally 
(i.e., through a dynamic programming approach). For example, you can break the problem into two 
subproblems, creating five buckets for FICO scores ranging from 0 to 600 and five buckets for FICO 
scores ranging from 600 to 850. Refer to this page for more context behind a likelihood function. 
This page may also be helpful for background on dynamic programming. """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Pre-existing functions

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

def quantize_fico_scores_with_risk(fico_scores, borrower_data, num_buckets, model, scaler, method='mse', score_range=(0, 850)):
    fico_scores.sort()
    total_scores = len(fico_scores)
    bucket_boundaries = np.linspace(score_range[0], score_range[1], num=num_buckets + 1)
    representatives = []
    bucket_pds = []

    for i in range(num_buckets):
        start = bucket_boundaries[i]
        end = bucket_boundaries[i + 1]
        bucket_indices = [j for j, score in enumerate(fico_scores) if start <= score < end]
        bucket_scores = [fico_scores[j] for j in bucket_indices]
        bucket_borrowers = borrower_data.iloc[bucket_indices]
        
        if bucket_scores:
            representative = np.mean(bucket_scores)
        else:
            representative = 0
        representatives.append(representative)

        # Calculate average PD for the bucket
        bucket_pd = np.mean([predict_expected_loss(model, scaler, borrower_data.iloc[j].tolist(), recovery_rate=0.1) 
                             for j in bucket_indices])
        bucket_pds.append(bucket_pd)

    # Assign ratings based on PDs (lower PD => better rating)
    sorted_pds = sorted(bucket_pds)
    bucket_ratings = [sorted_pds.index(pd) + 1 for pd in bucket_pds]

    return list(bucket_boundaries), bucket_ratings, representatives

def main():
    # Example usage
    pd.set_option("display.max_columns", None)
    path = 'Task 3 and 4 Loan Data.csv'
    df = load_and_preprocess_data(path)
    model, scaler = train_model(df)

    fico_scores = df['fico_score'].tolist()
    borrower_data = df.drop(columns=['customer_id', 'default'])

    num_buckets = 3
    bucket_boundaries, bucket_ratings, representatives = quantize_fico_scores_with_risk(
        fico_scores, borrower_data, num_buckets, model, scaler, method="mse"
    )

    print("Bucket Boundaries:", bucket_boundaries)
    print("Bucket Ratings:", bucket_ratings)
    print("Representatives:", representatives)

if __name__ == "__main__":
    main()
# Dataset

This project uses the **Telco Customer Churn** dataset, a widely used
public dataset for customer churn prediction.

## Option 1: Automatic (already included)

The file `Telco-Customer-Churn.csv` is included in this folder, so
`python main.py` will work immediately with no extra steps.

## Option 2: Download it yourself

If you ever need a fresh copy, you can download it from either source below
and place it in this folder with the exact name `Telco-Customer-Churn.csv`:

1. **Kaggle**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. **IBM Sample Data Sets (GitHub mirror)**:
   `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

## Dataset Description

Each row represents one customer of a fictional telecom company. Columns include:

- **customerID** — unique customer identifier (dropped during preprocessing)
- **gender, SeniorCitizen, Partner, Dependents** — customer demographics
- **tenure** — number of months the customer has stayed with the company
- **PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
  DeviceProtection, TechSupport, StreamingTV, StreamingMovies** — subscribed services
- **Contract, PaperlessBilling, PaymentMethod** — account information
- **MonthlyCharges, TotalCharges** — billing information
- **Churn** — target column (`Yes`/`No`) indicating if the customer left the company

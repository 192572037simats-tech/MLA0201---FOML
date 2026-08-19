import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Create dataset
n = 300

credit_score = np.random.randint(300, 851, n)
income = np.random.randint(20000, 150001, n)
debt = np.random.randint(1000, 80001, n)
age = np.random.randint(21, 61, n)

data = pd.DataFrame({
    "CreditScore": credit_score,
    "Income": income,
    "Debt": debt,
    "Age": age
})

# Classification rule
data["Class"] = np.where(
    (data["CreditScore"] >= 650) &
    (data["Income"] >= 40000) &
    (data["Debt"] < 50000),
    "Good", "Poor"
)

X = data[["CreditScore", "Income", "Debt", "Age"]]
y = data["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Credit Score Classification")
print("---------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Prediction for first 5 test records:")
print(y_pred[:5])

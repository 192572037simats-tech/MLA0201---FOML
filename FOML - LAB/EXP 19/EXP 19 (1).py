import numpy as np
from sklearn.naive_bayes import GaussianNB

# Training data
X = np.array([
    [60000, 30, 750, 200000],
    [50000, 35, 700, 150000],
    [45000, 28, 680, 100000],
    [30000, 40, 550, 250000],
    [25000, 45, 500, 300000],
    [20000, 50, 450, 350000]
])

y = [
    "Approved",
    "Approved",
    "Approved",
    "Rejected",
    "Rejected",
    "Rejected"
]

model = GaussianNB()
model.fit(X, y)

# Input
income = float(input("Enter Annual Income: "))
age = int(input("Enter Age: "))
credit = float(input("Enter Credit Score: "))
loan = float(input("Enter Loan Amount: "))

prediction = model.predict([
    [income, age, credit, loan]
])

print("\nLoan Prediction:", prediction[0])
6000

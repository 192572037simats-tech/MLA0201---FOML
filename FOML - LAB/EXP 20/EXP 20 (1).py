import numpy as np
from sklearn.linear_model import LinearRegression

# Previous sales data
months = np.array([
    [1], [2], [3], [4], [5], [6],
    [7], [8], [9], [10], [11], [12]
])

sales = np.array([
    50000, 55000, 60000, 65000,
    70000, 75000, 80000, 85000,
    90000, 95000, 100000, 105000
])

model = LinearRegression()
model.fit(months, sales)

# Input
month = int(input("Enter Future Month Number: "))

prediction = model.predict([[month]])

print("\nPredicted Sales: Rs.", round(prediction[0], 2))
15
1

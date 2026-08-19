import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([
    [500, 1, 10],
    [800, 2, 8],
    [1000, 2, 6],
    [1500, 3, 5],
    [2000, 3, 3],
    [2500, 4, 2]
])

y = np.array([
    2500000,
    4000000,
    5000000,
    7500000,
    10000000,
    12500000
])

model = LinearRegression()
model.fit(X, y)

# Input
area = float(input("Enter House Area (sq.ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
age = float(input("Enter House Age: "))

price = model.predict([[area, bedrooms, age]])

print("\nPredicted House Price: Rs.", round(price[0], 2))

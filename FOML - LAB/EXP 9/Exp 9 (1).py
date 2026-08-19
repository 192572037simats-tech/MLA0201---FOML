import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Input data
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])

# -------------------------
# Linear Regression
# -------------------------

linear = LinearRegression()
linear.fit(x, y)

linear_output = linear.predict(x)

# -------------------------
# Polynomial Regression
# -------------------------

poly = PolynomialFeatures(degree=2)

x_poly = poly.fit_transform(x)

polynomial = LinearRegression()
polynomial.fit(x_poly, y)

polynomial_output = polynomial.predict(x_poly)

# -------------------------
# Display Results
# -------------------------

print("LINEAR REGRESSION")
print("=================")

print("Predicted Values:")
print(np.round(linear_output, 2))

print("R2 Score:")
print(round(r2_score(y, linear_output), 2))

print("\nPOLYNOMIAL REGRESSION")
print("=====================")

print("Predicted Values:")
print(np.round(polynomial_output, 2))

print("R2 Score:")
print(round(r2_score(y, polynomial_output), 2))

# -------------------------
# Comparison
# -------------------------

print("\nCOMPARISON")
print("===========")

if r2_score(y, polynomial_output) > r2_score(y, linear_output):
    print("Polynomial Regression gives better performance.")
else:
    print("Linear Regression gives better performance.")

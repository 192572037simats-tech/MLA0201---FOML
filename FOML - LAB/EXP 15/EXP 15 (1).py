from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

iris = load_iris()

X = iris.data
y = iris.target

model = GaussianNB()
model.fit(X, y)

print("Enter Iris Flower Measurements")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

prediction = model.predict([[sl, sw, pl, pw]])

print("\nPredicted Flower:", iris.target_names[prediction[0]])

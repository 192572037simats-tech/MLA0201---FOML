import numpy as np
from sklearn.mixture import GaussianMixture

X = np.array([
    [1], [1.5], [2], [2.5], [3],
    [8], [8.5], [9], [9.5], [10]
])

model = GaussianMixture(
    n_components=2,
    random_state=42
)

model.fit(X)

labels = model.predict(X)

print("Data:")
print(X.flatten())

print("\nCluster Labels:")
print(labels)

print("\nCluster Means:")
print(model.means_.flatten())

print("\nCluster Weights:")
print(model.weights_)

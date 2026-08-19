import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Training data
X = np.array([
    [2, 32, 3000, 8],
    [3, 64, 4000, 12],
    [4, 64, 4500, 16],
    [6, 128, 5000, 48],
    [8, 256, 5000, 64],
    [12, 512, 6000, 108]
])

y = ["Low", "Low", "Medium", "Medium", "High", "High"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Input
ram = int(input("Enter RAM (GB): "))
storage = int(input("Enter Storage (GB): "))
battery = int(input("Enter Battery (mAh): "))
camera = int(input("Enter Camera (MP): "))

prediction = model.predict([[ram, storage, battery, camera]])

print("\nPredicted Mobile Price Category:", prediction[0])

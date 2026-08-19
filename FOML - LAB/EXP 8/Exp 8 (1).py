from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample heart disease data
X = [
    [63, 145, 233, 150, 2],
    [37, 130, 250, 187, 0],
    [41, 130, 204, 172, 0],
    [56, 120, 236, 178, 0],
    [57, 140, 192, 148, 1],
    [57, 140, 192, 148, 1],
    [44, 120, 263, 173, 0],
    [52, 172, 199, 162, 0],
    [57, 150, 168, 174, 1],
    [54, 140, 239, 160, 0],
    [48, 130, 275, 139, 0],
    [49, 130, 266, 171, 0],
    [64, 110, 211, 144, 1],
    [58, 150, 283, 162, 0],
    [50, 120, 219, 158, 0],
    [58, 120, 284, 160, 1],
    [66, 150, 226, 114, 1],
    [43, 150, 247, 171, 0],
    [69, 140, 239, 151, 1],
    [59, 135, 234, 161, 0]
]

# Target values
y = [
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    0, 0, 1, 0, 0,
    1, 1, 0, 1, 0
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create K-NN model
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

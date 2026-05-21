import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
X = np.load("dataset_X.npy")
y = np.load("dataset_y.npy")

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LogisticRegression(
    max_iter=200,
)

model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

# random guess is 1/34 ≈ 2.94% 
# model predicts which tile type to discard from a set of 34 possible tiles

# 5000 samples: 20.4% accuracy
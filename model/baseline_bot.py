import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

start_time = time.perf_counter()


# load dataset
X = np.load("dataset_X.npy")
y = np.load("dataset_y.npy")

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LogisticRegression(
    max_iter=1000,
)

model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time: {total_time:.4f} seconds")

# random guess is 1/34 ≈ 2.94% 
# model predicts which tile type to discard from a set of 34 possible tiles

# 5000 samples: .204 accuracy
# 50000 samples: .1584 accuracy
# 1151 seconds = 19.18 minutes
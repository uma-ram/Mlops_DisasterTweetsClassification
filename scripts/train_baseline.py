import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

from scripts.preprocessing import load_and_split_data, vectorize_text

# Load and split
X_train, X_test, y_train, y_test = load_and_split_data()

# Vectorize
X_train_vec, X_test_vec, tfidf = vectorize_text(X_train, X_test)

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/baseline_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

import joblib

# Load model & vectorizer
clf = joblib.load("models/baseline_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Use for prediction
text = ["happy at the airport!"]
X = tfidf.transform(text)
pred = clf.predict(X)
print("Predicted label:", pred[0])

import mlflow
import mlflow.sklearn
from scripts.preprocessing import load_and_split_data, vectorize_text

# Start tracking
mlflow.set_experiment("disaster_tweets")

with mlflow.start_run():
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # Train model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log params, metrics, model
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

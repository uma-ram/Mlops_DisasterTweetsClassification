import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from scripts.preprocessing import load_and_split_data, vectorize_text

# Step 1: Set experiment
mlflow.set_experiment("Model_Comparison_Disaster_Tweets")

# Step 2: Load & preprocess data
X_train, X_test, y_train, y_test = load_and_split_data()
X_train_vec, X_test_vec, _ = vectorize_text(X_train, X_test)

# Step 3: Define models
models = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=200),
    "SVM": LinearSVC(C=1.0),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
}

# Step 4: Train & log each model
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model", name)
        
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("model_name", name)
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

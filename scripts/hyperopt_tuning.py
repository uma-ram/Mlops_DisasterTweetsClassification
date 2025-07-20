import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scripts.preprocessing import load_and_split_data, vectorize_text

# Set the experiment name (will create if doesn't exist)
mlflow.set_experiment("Hyperopt_Tuning_Trials")

# Load and vectorize data
X_train, X_test, y_train, y_test = load_and_split_data()
X_train_vec, X_test_vec, _ = vectorize_text(X_train, X_test)

def objective(params):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model", "LogisticRegression")
        mlflow.set_tag("tuning", "hyperopt")

        clf = LogisticRegression(**params)
        clf.fit(X_train_vec, y_train)

        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        return {'loss': -f1, 'status': STATUS_OK}

# Define hyperparameter search space
space = {
    'C': hp.loguniform('C', -4, 2),
    'max_iter': hp.choice('max_iter', [100, 200, 300]),
    'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
}

trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)

print("Best hyperparameters:", best)


import json
import os
from datetime import datetime
from pathlib import Path
 
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, brier_score_loss
 
from data_preprocessing import load_data, get_X_y, RISK_THRESHOLD
 
# Work from the project root no matter where the script is launched from
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
 
RANDOM_STATE = 42
 
 
def build_pipeline(model, X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])
 
 
if __name__ == "__main__":
    df = load_data()
    X, y = get_X_y(df)
 
    # Stratified split, fixed random_state; test set stays untouched until evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)
 
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
 
    os.makedirs("models", exist_ok=True)
 
    brier = {}
    for name, model in models.items():
        pipe = build_pipeline(model, X_train)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, f"models/{name}.joblib")
 
        print(f"\n=== {name} ===")
        print(classification_report(
            y_test, pipe.predict(X_test),
            target_names=["Not At Risk", "At Risk"]
        ))
        # Brier score: lower is better; measures how good the probabilities are
        brier[name] = round(
            brier_score_loss(y_test, pipe.predict_proba(X_test)[:, 1]), 4
        )
        print(f"Brier score (probability quality, lower is better): {brier[name]}")
 
    # Save the test split so the evaluation step uses the same data
    joblib.dump((X_test, y_test), "models/test_data.joblib")
 
    # Save training details so the app can warn about version mismatches
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "risk_threshold": RISK_THRESHOLD,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": X.columns.tolist(),
        "brier_score": brier,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
 
    print("\nModels, test data and metadata saved in the models/ folder.")